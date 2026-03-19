"""Parse the NHTSA FLAT_CMPL.txt bulk complaints file and load into nhtsa_bulk.db."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Index,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import BASE_DIR

logger = logging.getLogger(__name__)

BULK_DB_PATH = BASE_DIR / "nhtsa_bulk.db"
BULK_DB_URL = f"sqlite:///{BULK_DB_PATH}"
FLAT_CMPL_PATH = BASE_DIR / "dataset" / "FLAT_CMPL.txt"

BATCH_SIZE = 10_000


class BulkBase(DeclarativeBase):
    pass


class NHTSAComplaint(BulkBase):
    __tablename__ = "nhtsa_complaints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cmpl_id = Column(Integer, nullable=False, index=True)
    make = Column(String(100), nullable=False, index=True)
    model = Column(String(100), nullable=False, index=True)
    year = Column(Integer, nullable=False, index=True)
    component_raw = Column(String(500))
    system = Column(String(50), index=True)
    narrative = Column(Text)
    mileage = Column(Integer)
    crash = Column(Boolean, default=False)
    fire = Column(Boolean, default=False)
    injured = Column(Integer, default=0)
    deaths = Column(Integer, default=0)
    date_filed = Column(Date)

    __table_args__ = (
        Index("ix_make_model_year", "make", "model", "year"),
        Index("ix_system_year", "system", "year"),
    )


NHTSA_COMPONENT_TO_SYSTEM = {
    "engine": "Engine",
    "power train": "Transmission",
    "automatic transmission": "Transmission",
    "manual transmission": "Transmission",
    "clutch": "Transmission",
    "electrical": "Electrical",
    "electronic stability control": "Electrical",
    "lights": "Electrical",
    "wiring": "Electrical",
    "suspension": "Suspension",
    "wheels": "Suspension",
    "tires": "Suspension",
    "service brakes": "Brakes",
    "parking brake": "Brakes",
    "brake": "Brakes",
    "air bags": "Interior",
    "seat belts": "Interior",
    "seats": "Interior",
    "interior": "Interior",
    "child seat": "Interior",
    "body": "Body/Paint",
    "structure": "Body/Paint",
    "exterior": "Body/Paint",
    "latches": "Body/Paint",
    "paint": "Body/Paint",
    "visibility": "Body/Paint",
    "air conditioning": "HVAC",
    "heater": "HVAC",
    "steering": "Steering",
    "fuel system": "Fuel System",
    "fuel": "Fuel System",
    "exhaust": "Exhaust",
    "emission": "Exhaust",
    "cooling": "Cooling",
    "radiator": "Cooling",
}


def _map_component_to_system(component_raw: str) -> str:
    """Map NHTSA's component description to one of the 12 standard vehicle systems."""
    if not component_raw:
        return "Other"
    lower = component_raw.lower()
    for keyword, system in NHTSA_COMPONENT_TO_SYSTEM.items():
        if keyword in lower:
            return system
    return "Other"


def _parse_bool(val: str) -> bool:
    return val.strip().upper() == "Y"


def _parse_int(val: str) -> int | None:
    val = val.strip()
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        nums = re.findall(r"\d+", val)
        return int(nums[0]) if nums else None


def _parse_date(val: str) -> datetime | None:
    val = val.strip()
    if not val or len(val) < 8:
        return None
    try:
        return datetime.strptime(val[:8], "%Y%m%d").date()
    except ValueError:
        return None


_bulk_engine = None
_bulk_engine_lock = __import__("threading").Lock()


def _get_bulk_engine():
    global _bulk_engine
    if _bulk_engine is not None:
        return _bulk_engine
    with _bulk_engine_lock:
        if _bulk_engine is not None:
            return _bulk_engine
        _bulk_engine = create_engine(
            BULK_DB_URL, echo=False,
            connect_args={"check_same_thread": False},
        )
        return _bulk_engine


def get_bulk_session() -> Session:
    engine = _get_bulk_engine()
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


def load_flat_cmpl(filepath: str | Path | None = None, drop_existing: bool = True):
    """Parse FLAT_CMPL.txt and load all complaints into nhtsa_bulk.db.

    The NHTSA flat file is tab-delimited with these key column positions:
    0: row number, 1: CMPLID, 2: MFR_NAME, 3: MAKE, 4: MODEL, 5: YEAR,
    6: CRASH (Y/N), 7: FAILDATE, 8: FIRE (Y/N), 9: INJURED count, 10: DEATHS count,
    11: COMPDESC, 12: CITY, 13: STATE, 14: VIN prefix,
    15: DATEA (date added), 16: date updated,
    19: CDESCR (narrative),
    44: MILES (odometer reading)
    """
    filepath = Path(filepath) if filepath else FLAT_CMPL_PATH
    if not filepath.exists():
        raise FileNotFoundError(
            f"Bulk complaints file not found: {filepath}\n"
            "Download from https://www-odi.nhtsa.dot.gov/downloads/folders/Complaints/FLAT_CMPL.zip"
        )

    engine = _get_bulk_engine()
    if drop_existing:
        BulkBase.metadata.drop_all(engine)
    BulkBase.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    total = 0
    skipped = 0
    batch: list[NHTSAComplaint] = []

    logger.info("Loading NHTSA bulk complaints from %s ...", filepath)

    with open(filepath, encoding="latin-1") as f:
        for line_num, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 20:
                skipped += 1
                continue

            cmpl_id = _parse_int(fields[1])
            if cmpl_id is None:
                skipped += 1
                continue

            make = fields[3].strip().upper()
            model_name = fields[4].strip().upper()
            year = _parse_int(fields[5])
            if not make or not model_name or year is None:
                skipped += 1
                continue

            component_raw = fields[11].strip()
            system = _map_component_to_system(component_raw)
            narrative = fields[19].strip() if len(fields) > 19 else ""
            mileage = _parse_int(fields[44]) if len(fields) > 44 else None

            crash = _parse_bool(fields[6])
            fire = _parse_bool(fields[8])
            injured = _parse_int(fields[9]) or 0
            deaths = _parse_int(fields[10]) or 0

            date_filed = _parse_date(fields[15])

            complaint = NHTSAComplaint(
                cmpl_id=cmpl_id,
                make=make,
                model=model_name,
                year=year,
                component_raw=component_raw,
                system=system,
                narrative=narrative,
                mileage=mileage,
                crash=crash,
                fire=fire,
                injured=injured,
                deaths=deaths,
                date_filed=date_filed,
            )
            batch.append(complaint)
            total += 1

            if len(batch) >= BATCH_SIZE:
                session.bulk_save_objects(batch)
                session.commit()
                batch.clear()
                if total % 100_000 == 0:
                    logger.info("  ... loaded %d complaints", total)

    if batch:
        session.bulk_save_objects(batch)
        session.commit()

    session.close()
    logger.info(
        "Bulk load complete: %d complaints loaded, %d rows skipped. DB at %s",
        total, skipped, BULK_DB_PATH,
    )
    return total
