from datetime import datetime, timedelta

def generate_schedule(tasks):
    day = datetime.now().date()
    current = datetime.strptime("10:00", "%H:%M")
    out = []

    for t in tasks:
        start = current.strftime("%I:%M %p")
        end_time = current + timedelta(hours=min(t["hours"], 2))
        end = end_time.strftime("%I:%M %p")
        out.append({
            "day": day.isoformat(),
            "start": start,
            "end": end,
            "task": t["task"],
            "assignment": t["assignment"]
        })
        current = end_time + timedelta(minutes=10)

    return out
