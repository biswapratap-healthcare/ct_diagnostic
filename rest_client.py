import json
import requests


def update_in_progress_state(sid, state):
    r = requests.post("http://localhost:5000/update_in_progress",
                      data={'study_instance_id': sid, 'in_progress': state})
    d = json.loads(r.text)
    return d['status']


def update_progress_percent(sid, percent):
    r = requests.post("http://localhost:5000/update_percent",
                      data={'study_instance_id': sid, 'percent': percent})
    d = json.loads(r.text)
    return d['status']


def save_result(sid, result_zip_path):
    r = requests.post("http://localhost:5000/save_result",
                      data={'study_instance_id': sid},
                      files={'zip_file': open(result_zip_path, 'rb')})
    d = json.loads(r.text)
    return d['status']
