import os
import shutil
import tempfile
import threading
import zipfile
from zipfile import ZipFile

from waitress import serve
from flask_cors import CORS
from flask import Flask, send_file

from flask_restplus import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

from driver import generate_report, store_and_verify_file
from report_assembler import assemble_report
from utils import read_progress, write_progress


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    api = Api(
        app,
        version='1.0.0',
        title='CT Predictor App',
        description='CT Predictor App',
        default='CT Predictor App',
        default_label=''
    )

    CORS(app)

    generate_report_parser = reqparse.RequestParser()
    generate_report_parser.add_argument('zip_file',
                                        location='files',
                                        type=FileStorage,
                                        help='The zip of CT files',
                                        required=True)

    @api.route('/generate_report')
    @api.expect(generate_report_parser)
    class GenerateReportService(Resource):
        @api.expect(generate_report_parser)
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                args = generate_report_parser.parse_args()
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404
            try:
                file_from_request = args['zip_file']
                job_id = study_instance_id = output_dir = file_from_request.filename[:-4]
                if os.path.exists(output_dir):
                    assemble_report(output_dir)
                else:
                    os.makedirs(output_dir)
                    write_progress(job_id, "5")
                    file_dir = tempfile.mkdtemp()
                    work_dir = tempfile.mkdtemp()
                    ret, file_path = store_and_verify_file(file_from_request, work_dir=file_dir)
                    if ret == 0:
                        with ZipFile(file_path, 'r') as zipObj:
                            zipObj.extractall(work_dir)
                        write_progress(job_id, "10")
                        t = threading.Thread(target=generate_report, args=(study_instance_id, work_dir, output_dir))
                        t.start()
                        shutil.rmtree(file_dir)
                        rv = dict()
                        rv['status'] = "Started"
                        rv['job_id'] = job_id
                        return rv, 200
                    else:
                        rv = dict()
                        rv['status'] = str(file_path)
                        return rv, 404
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    health_check_parser = reqparse.RequestParser()
    health_check_parser.add_argument('var',
                                     type=int,
                                     help='dummy variable',
                                     required=True)

    @api.route('/health_check')
    @api.expect(health_check_parser)
    class HealthCheckService(Resource):
        @api.expect(health_check_parser)
        @api.doc(responses={"response": 'json'})
        def post(self):
            try:
                args = health_check_parser.parse_args()
            except Exception as e:
                rv = dict()
                rv['health'] = str(e)
                return rv, 404
            rv = dict()
            rv['health'] = "good"
            return rv, 200

    get_progress_parser = reqparse.RequestParser()
    get_progress_parser.add_argument('job_id',
                                     type=str,
                                     help='job_id',
                                     required=True)

    @api.route('/get_progress')
    @api.expect(get_progress_parser)
    class GetProgressService(Resource):
        @api.expect(get_progress_parser)
        @api.doc(responses={"response": 'json'})
        def get(self):
            try:
                args = get_progress_parser.parse_args()
                job_id = args['job_id']
                percent = read_progress(job_id)
                rv = dict()
                rv['percent'] = percent
                return rv, 200
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    get_report_parser = reqparse.RequestParser()
    get_report_parser.add_argument('job_id',
                                   type=str,
                                   help='job_id',
                                   required=True)

    @api.route('/get_report')
    @api.expect(get_report_parser)
    class GetReportService(Resource):
        @api.expect(get_report_parser)
        def get(self):
            try:
                args = get_report_parser.parse_args()
                job_id = args['job_id']
                zip_folder = zipfile.ZipFile('result.zip', 'w', compression=zipfile.ZIP_STORED)
                for file in os.listdir(job_id):
                    zip_folder.write(job_id + '/' + file)
                zip_folder.close()

                return send_file('result.zip',
                                 mimetype='zip',
                                 attachment_filename='result.zip',
                                 as_attachment=True)
            except Exception as e:
                rv = dict()
                rv['status'] = str(e)
                return rv, 404

    return app


if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=5001)
