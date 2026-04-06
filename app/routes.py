from flask import Blueprint, render_template, request, Response, stream_with_context, jsonify

from .tutor import stream_response
from .math_tools import generate_practice_problem

bp = Blueprint("main", __name__)


@bp.route("/")
def index():
    return render_template("index.html")


@bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    messages = data.get("messages", [])
    mode = data.get("mode", "chat")
    equation = data.get("equation") or None

    def generate():
        yield from stream_response(messages, mode, equation)

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@bp.route("/new-problem", methods=["POST"])
def new_problem():
    data = request.get_json(force=True)
    difficulty = data.get("difficulty", "easy")
    topic = data.get("topic", "linear")
    problem = generate_practice_problem(difficulty, topic)
    return jsonify(problem)
