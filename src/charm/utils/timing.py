def format_time(running_time: float) -> str:
    """Format a time in seconds as HH:MM:SS."""
    hrs = int(running_time // 3600)
    mts = int((running_time - hrs * 3600) // 60)
    sec = round(running_time - hrs * 3600 - mts * 60)

    return f"{hrs:02d}:{mts:02d}:{sec:02d}"
