"""
koffie/strategy/models/atr_analysis.py

Zone width / ATR ratio analysis for research and diagnostics.
Helps understand which zone types (by volatility) are most profitable.

Imported from keithkulecho/spreadsheet_bot diagnose_zone_width_atr.py logic.
"""

from enum import Enum
from typing import Optional, List, Dict, Any

import pandas as pd

from koffie.strategy.atr import atr_series


class ZoneWidthATRBucket(Enum):
    """Zone classification by width/ATR ratio."""
    NARROW = "NARROW"      # width/ATR < 0.5:  tight zones in low volatility
    NORMAL = "NORMAL"      # 0.5 <= width/ATR <= 2.0:  typical zones
    WIDE = "WIDE"          # width/ATR > 2.0:  expansive zones in high volatility


def calculate_width_atr_ratio(
    zone_width: float,
    atr_value: float,
) -> Optional[float]:
    """Calculate the ratio of zone width to ATR at a specific moment.
    
    Args:
        zone_width: High - Low of the zone
        atr_value: ATR(14) at the time the zone was formed
    
    Returns:
        float: width / atr_value, or None if atr_value is invalid
    """
    if atr_value is None or atr_value <= 0:
        return None
    return zone_width / atr_value


def bucket_by_width_atr_ratio(ratio: Optional[float]) -> ZoneWidthATRBucket:
    """Classify a zone based on its width/ATR ratio.
    
    Args:
        ratio: width / ATR(14) at zone formation
    
    Returns:
        ZoneWidthATRBucket: NARROW, NORMAL, or WIDE
    """
    if ratio is None:
        return None
    if ratio < 0.5:
        return ZoneWidthATRBucket.NARROW
    if ratio > 2.0:
        return ZoneWidthATRBucket.WIDE
    return ZoneWidthATRBucket.NORMAL


def analyze_zones_by_width_atr(
    zones_data: List[Dict[str, Any]],
    atr_series_data: pd.Series,
) -> pd.DataFrame:
    """Analyze a list of zones by their width/ATR ratio at formation time.
    
    Args:
        zones_data: List of dicts with keys:
            - 'formed_at': timestamp the zone was created
            - 'price_top': float
            - 'price_bottom': float
            - (optional) 'zone_type': DEMAND or SUPPLY
        atr_series_data: pd.Series indexed by timestamp, values are ATR(14)
    
    Returns:
        pd.DataFrame with columns:
            - formed_at
            - price_top, price_bottom
            - width
            - atr_at_formation
            - width_atr_ratio
            - bucket (NARROW/NORMAL/WIDE)
    """
    rows = []
    for zone in zones_data:
        formed_at = zone.get('formed_at')
        price_top = zone.get('price_top')
        price_bottom = zone.get('price_bottom')
        
        if formed_at is None or price_top is None or price_bottom is None:
            continue
        
        width = price_top - price_bottom
        
        # Get ATR value at the moment this zone was formed
        # Use asof() to get the most recent known value up to that timestamp
        atr_at_formation = atr_series_data.asof(formed_at)
        
        ratio = calculate_width_atr_ratio(width, atr_at_formation)
        bucket = bucket_by_width_atr_ratio(ratio)
        
        row = {
            'formed_at': formed_at,
            'price_top': price_top,
            'price_bottom': price_bottom,
            'width': width,
            'atr_at_formation': atr_at_formation,
            'width_atr_ratio': ratio,
            'bucket': bucket.value if bucket else None,
        }
        
        # Add optional fields
        if 'zone_type' in zone:
            row['zone_type'] = zone['zone_type']
        
        rows.append(row)
    
    return pd.DataFrame(rows)
