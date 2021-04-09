from waitress import serve
from flask_cors import CORS
from flask import Flask, send_file

from flask_restplus import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage

from driver import generate_report


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
                rv['diagnosis'] = str(e)
                return rv, 404
            d = generate_report(args, app.logger)
            return d, 200

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

    return app


if __name__ == "__main__":
    serve(create_app(), host='0.0.0.0', port=5000)
