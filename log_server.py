from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__)

# 配置
LOG_FILE = 'training_log.txt'
HTML_FILE = 'log_viewer.html'
PORT = 5000


@app.route('/')
def index():
    return send_from_directory('.', HTML_FILE)


@app.route('/training_log.txt')
def log_file():
    if os.path.exists(LOG_FILE):
        return send_from_directory('.', LOG_FILE)
    return "日志文件不存在", 404


@app.route('/api/log_info')
def log_info():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return jsonify({
            'exists': True,
            'line_count': len(lines),
            'file_size': os.path.getsize(LOG_FILE),
            'last_modified': os.path.getmtime(LOG_FILE)
        })
    return jsonify({
        'exists': False,
        'line_count': 0,
        'file_size': 0,
        'last_modified': 0
    })


if __name__ == '__main__':
    print("=" * 50)
    print("训练日志服务器启动中...")
    print("=" * 50)
    print(f"本地访问: http://localhost:{PORT}")
    print(f"服务器访问: http://服务器IP:{PORT}")
    print("=" * 50)
    print("按 Ctrl+C 停止服务器")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=PORT, debug=False)
