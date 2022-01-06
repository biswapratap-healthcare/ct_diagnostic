import json
import requests


def update_in_progress_state(sid, state):
    r = requests.post("http://localhost:7000/update_in_progress",
                      data={'study_instance_id': sid, 'in_progress': state})
    d = json.loads(r.text)
    return d['status']


def update_progress_percent(sid, percent):
    r = requests.post("http://localhost:7000/update_percent",
                      data={'study_instance_id': sid, 'percent': percent})
    d = json.loads(r.text)
    return d['status']


def save_result(sid, result_zip_path):
    r = requests.post("http://localhost:7000/save_result",
                      data={'study_instance_id': sid, 'file': result_zip_path})
    d = json.loads(r.text)
    return d['status']
