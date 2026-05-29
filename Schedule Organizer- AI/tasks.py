import re

def extract_subtasks(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines

def normalize_task_name(t):
    return t.title()
