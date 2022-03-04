import json
import requests


def update_in_progress_state(sid, state):
    r = requests.post("http://localhost:9800/__update_in_progress__",
                      data={'study_instance_id': sid, 'in_progress': state})
    d = json.loads(r.text)
    return d['status']


def update_progress_percent(sid, percent):
    r = requests.post("http://localhost:9800/__update_percent__",
                      data={'study_instance_id': sid, 'percent': percent})
    d = json.loads(r.text)
    return d['status']


def save_result(sid, result_zip_path):
    r = requests.post("http://localhost:9800/__save_result__",
                      data={'study_instance_id': sid, 'file': result_zip_path})
    d = json.loads(r.text)
    return d['status']
