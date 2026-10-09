"""
tools/diagnose_zone_width_atr.py

Diagnostic script: Analyze zone profitability by width/ATR ratio.

Usage:
    python -m tools.diagnose_zone_width_atr <historical_data_csv> <trades_csv>

Example:
    python -m tools.diagnose_zone_width_atr data/historical.csv results/trades.csv

Imported from keithkulecho/spreadsheet_bot diagnose_zone_width_atr.py.

This script:
1. Computes ATR(14) from historical OHLC data
2. Calculates zone width / ATR ratio at zone formation time
3. Buckets zones as NARROW (<0.5), NORMAL (0.5-2.0), WIDE (>2.0)
4. Matches historical trades to nearest prior zone
5. Reports win rate and expectancy (R-multiple) by bucket

This helps research which zone types (by volatility regime) are most profitable.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent directory to path so imports work when run as module
sys.path.insert(0, str(Path(__file__).parent.parent))

from koffie.strategy.atr import atr_series
from koffie.strategy.models.atr_analysis import (
    analyze_zones_by_width_atr,
    bucket_by_width_atr_ratio,
    ZoneWidthATRBucket,
)


def load_historical_data(csv_path: str) -> pd.DataFrame:
    """Load historical OHLC data from CSV.
    
    Expected columns: time, open, high, low, close
    or: timestamp, open, high, low, close
    """
    df = pd.read_csv(csv_path)
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Find timestamp column
    time_col = None
    for col in ['time', 'timestamp', 'datetime']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No timestamp column found. Expected one of: time, timestamp, datetime")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    df = df.sort_index()
    
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if df[['open', 'high', 'low', 'close']].isna().any().any():
        raise ValueError("OHLC data contains NaN values")
    
    return df


def load_zones(zone_csv_path: str) -> list:
    """Load zones from CSV.
    
    Expected columns: formed_at, price_top, price_bottom (and optionally zone_type)
    """
    try:
        df = pd.read_csv(zone_csv_path)
    except FileNotFoundError:
        print(f"Warning: Zone file not found: {zone_csv_path}")
        return []
    
    df.columns = [c.lower().strip() for c in df.columns]
    df['formed_at'] = pd.to_datetime(df['formed_at'])
    
    zones = []
    for _, row in df.iterrows():
        zone = {
            'formed_at': row['formed_at'],
            'price_top': float(row['price_top']),
            'price_bottom': float(row['price_bottom']),
        }
        if 'zone_type' in df.columns:
            zone['zone_type'] = row['zone_type']
        zones.append(zone)
    
    return zones


def load_trades(trades_csv_path: str) -> pd.DataFrame:
    """Load trade results from CSV.
    
    Expected columns: entry_time, status, r_multiple (or pnl/risk)
    """
    df = pd.read_csv(trades_csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Normalize time column
    time_col = None
    for col in ['entry_time', 'entry_at', 'time', 'timestamp']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        raise ValueError(f"No entry time column found")
    
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    
    # Ensure we have status and r-multiple
    if 'status' not in df.columns:
        raise ValueError("Expected 'status' column (WIN/LOSS)")
    
    if 'r_multiple' not in df.columns:
        if 'pnl' in df.columns and 'risk' in df.columns:
            df['r_multiple'] = df['pnl'] / df['risk']
        else:
            raise ValueError("Expected 'r_multiple' column or both 'pnl' and 'risk'")
    
    df['entry_time'] = df[time_col]
    return df


def match_trades_to_zones(
    trades: pd.DataFrame,
    zones_analysis: pd.DataFrame,
    max_hours: float = 6.0,
) -> pd.DataFrame:
    """Match each trade to the nearest prior zone within max_hours.
    
    Args:
        trades: DataFrame with 'entry_time' column
        zones_analysis: DataFrame from analyze_zones_by_width_atr() with 'formed_at', 'bucket', 'width_atr_ratio'
        max_hours: Maximum time window to look back for a matching zone
    
    Returns:
        Original trades DataFrame with added 'zone_bucket' and 'width_atr_ratio' columns
    """
    trades = trades.copy()
    trades['zone_bucket'] = None
    trades['width_atr_ratio'] = None
    
    zones_sorted = zones_analysis.sort_values('formed_at').reset_index(drop=True)
    break_times = zones_sorted['formed_at'].values
    buckets_arr = zones_sorted['bucket'].values
    ratios_arr = zones_sorted['width_atr_ratio'].values
    
    for idx, row in trades.iterrows():
        entry_time = pd.Timestamp(row['entry_time'])
        ts64 = np.datetime64(entry_time)
        
        # Find all zones that formed before this trade
        mask = break_times <= ts64
        if not mask.any():
            continue
        
        # Get the most recent one
        pos = np.where(mask)[0][-1]
        
        # Check if within max_hours window
        gap_hours = (ts64 - break_times[pos]) / np.timedelta64(1, 'h')
        if gap_hours > max_hours:
            continue
        
        trades.at[idx, 'zone_bucket'] = buckets_arr[pos]
        trades.at[idx, 'width_atr_ratio'] = ratios_arr[pos]
    
    return trades


def main(historical_csv: str, zones_csv: str = None, trades_csv: str = None):
    """Main diagnostic function.
    
    Args:
        historical_csv: Path to historical OHLC data
        zones_csv: Path to zones CSV (optional, will try to infer)
        trades_csv: Path to trades CSV (optional, will try to infer)
    """
    print("Loading historical data...")
    df = load_historical_data(historical_csv)
    print(f"  Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    
    print("\nCalculating ATR(14)...")
    atr14 = atr_series(df, period=14)
    print(f"  ATR range: {atr14.min():.4f} to {atr14.max():.4f}")
    
    # Load zones if available
    if zones_csv is None:
        # Try to infer from directory
        possible_paths = [
            Path('zones.csv'),
            Path(historical_csv).parent / 'zones.csv',
        ]
        for p in possible_paths:
            if p.exists():
                zones_csv = str(p)
                break
    
    if zones_csv:
        print(f"\nLoading zones from {zones_csv}...")
        zones = load_zones(zones_csv)
        print(f"  Loaded {len(zones)} zones")
        
        zones_df = analyze_zones_by_width_atr(zones, atr14)
        zones_df = zones_df.dropna(subset=['width_atr_ratio'])
        
        print(f"\nZone width/ATR ratio distribution:")
        if len(zones_df) > 0:
            print(zones_df['width_atr_ratio'].describe().round(3).to_string())
            print(f"\nZone count by bucket:")
            print(zones_df['bucket'].value_counts())
    else:
        zones_df = None
        print("  (No zones CSV provided)")
    
    # Load trades if available
    if trades_csv is None:
        possible_paths = [
            Path('trades.csv'),
            Path(historical_csv).parent / 'trades.csv',
        ]
        for p in possible_paths:
            if p.exists():
                trades_csv = str(p)
                break
    
    if trades_csv and zones_df is not None:
        print(f"\nLoading trades from {trades_csv}...")
        trades = load_trades(trades_csv)
        print(f"  Loaded {len(trades)} trades")
        
        print("\nMatching trades to zones...")
        trades = match_trades_to_zones(trades, zones_df, max_hours=6.0)
        
        print(f"\n=== Trade quality by zone width/ATR bucket ===")
        matched = trades[trades['zone_bucket'].notna()]
        if len(matched) > 0:
            summary = matched.groupby('zone_bucket').agg(
                trades=('r_multiple', 'count'),
                win_rate=('status', lambda s: round(100 * (s == 'WIN').mean(), 2)),
                total_r=('r_multiple', lambda s: round(s.sum(), 2)),
                avg_r=('r_multiple', lambda s: round(s.mean(), 3)),
                min_r=('r_multiple', lambda s: round(s.min(), 3)),
                max_r=('r_multiple', lambda s: round(s.max(), 3)),
            )
            order = [ZoneWidthATRBucket.NARROW.value, ZoneWidthATRBucket.NORMAL.value, ZoneWidthATRBucket.WIDE.value]
            summary = summary.reindex([o for o in order if o in summary.index])
            print(summary.to_string())
            print(f"\nTotal matched trades: {len(matched)} / {len(trades)}")
        else:
            print("  (No trades matched to zones)")
    
    print("\n" + "="*60)
    print("Notes:")
    print("  - NARROW zones (width/ATR < 0.5) form in tight, low-volatility markets")
    print("  - WIDE zones (width/ATR > 2.0) form in loose, high-volatility markets")
    print("  - Compare win rate and avg R-multiple to see which thrive in which regime")
    print("  - Small sample sizes (<30 trades) should be treated as noise")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m tools.diagnose_zone_width_atr <historical_data.csv> [zones.csv] [trades.csv]")
        print("\nExample:")
        print("  python -m tools.diagnose_zone_width_atr data/historical.csv zones.csv trades.csv")
        sys.exit(1)
    
    hist_csv = sys.argv[1]
    zones_csv = sys.argv[2] if len(sys.argv) > 2 else None
    trades_csv = sys.argv[3] if len(sys.argv) > 3 else None
    
    main(hist_csv, zones_csv, trades_csv)
