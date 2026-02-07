from flask import Flask, render_template, request, jsonify
import random
import csv
import io

app = Flask(__name__)


# 模拟生成解的核心函数（您可以替换为实际的CVAE模型调用）
def generate_solutions(params, count):
    """
    根据输入参数生成符合约束的光纤结构参数解
    :param params: 6个光学特性参数的字典
    :param count: 要生成的解数量
    :return: 符合约束的解列表
    """
    solutions = []

    # 模拟生成解（您需要替换为实际的模型调用逻辑）
    for i in range(count):
        # 生成符合约束条件的随机数据（示例）
        n1 = round(random.uniform(0.018, 0.09), 4)
        n2 = round(random.uniform(-0.05, 0), 4)
        r8 = round(random.uniform(2.5, 8.5), 3)
        r9 = round(random.uniform(r8 + 1, 14.5), 3)  # 满足 r9 - r8 ≥ 1
        r10 = round(random.uniform(r9 + 0.5, 16.5), 3)  # 满足 r10 - r9 ≥ 0.5
        wl = round(random.uniform(1.5, 1.6), 2)

        solutions.append({
            'id': i + 1,
            'n1': n1,
            'n2': n2,
            'r8': r8,
            'r9': r9,
            'r10': r10,
            'wl': wl,
            'error': round(random.uniform(0.001, 0.01), 4)  # 模拟误差值
        })

    return solutions


@app.route('/')
def index():
    """渲染主页面"""
    return render_template('show.html')


@app.route('/generate', methods=['POST'])
def generate():
    """处理参数生成请求"""
    try:
        # 获取前端提交的参数
        data = request.json
        param1 = float(data.get('param1'))
        param2 = float(data.get('param2'))
        param3 = float(data.get('param3'))
        param4 = float(data.get('param4'))
        param5 = float(data.get('param5'))
        param6 = float(data.get('param6'))
        solution_count = int(data.get('solutionCount', 50))

        # 封装参数（供您调用其他函数使用）
        params = {
            'TE01-HE21': param1,
            'HE21-TM01': param2,
            'HE31-EH11': param3,
            'HE41-EH21': param4,
            'HE51-EH31': param5,
            'HE61-EH41': param6
        }

        # 调用生成函数（您可以替换为实际的模型调用）
        solutions = generate_solutions(params, solution_count)

        # 统计信息
        stats = {
            'target_count': solution_count,
            'valid_count': len(solutions),
            'pass_rate': f"{(len(solutions) / solution_count) * 100:.1f}%"
        }

        # 返回结果给前端
        return jsonify({
            'success': True,
            'stats': stats,
            'solutions': solutions
        })

    except Exception as e:
        # 异常处理
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/export', methods=['POST'])
def export_csv():
    """导出CSV文件"""
    try:
        data = request.json
        solutions = data.get('solutions', [])

        # 创建CSV文件
        output = io.StringIO()
        writer = csv.writer(output)

        # 写入表头
        writer.writerow(['序号', 'n1', 'n2', 'r8', 'r9', 'r10', 'wl', '误差值'])

        # 写入数据
        for sol in solutions:
            writer.writerow([
                sol['id'], sol['n1'], sol['n2'], sol['r8'],
                sol['r9'], sol['r10'], sol['wl'], sol['error']
            ])

        # 重置指针
        output.seek(0)

        # 返回CSV响应
        return jsonify({
            'success': True,
            'csv_data': output.getvalue()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


if __name__ == '__main__':
    # 确保templates文件夹中存在show.html文件
    # 调试模式运行，生产环境请关闭debug=True
    app.run(debug=True, host='127.0.0.1', port=5000)
