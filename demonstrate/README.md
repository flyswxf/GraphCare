# GraphCare Demonstrate 演示系统

本演示提供一个模拟 ICU 医生使用场景的前后端应用：
- 前端：拖拽/上传患者资料（JSON），展示心源性休克概率与推荐措施、相似病例处理方法频率；可在结果页继续补充自然语言信息并更新结果。
- 后端：提供两个接口：上传 JSON 推理接口、自然语言补充信息更新接口。内部使用启发式规则进行演示（非医疗用途）。

## 目录结构
```
demonstrate/
  ├── backend/
  │   ├── main.py              # FastAPI 后端
  │   └── requirements.txt     # 依赖
  └── frontend/
      ├── index.html           # 前端入口
      ├── style.css            # 样式
      └── app.js               # 前端逻辑
```

## 后端启动
1) 创建并激活虚拟环境（可选）：
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```

2) 安装依赖：
```
pip install -r demonstrate/backend/requirements.txt
```

3) 启动服务：
```
python -m uvicorn demonstrate.backend.main:app --host 0.0.0.0 --port 8000
```
启动成功后，健康检查地址：http://localhost:8000/api/health

## 前端启动
方式A（推荐）：使用 Python 简单静态服务器
```
cd demonstrate/frontend
python -m http.server 5173
```
然后在浏览器打开：http://localhost:5173/

方式B：直接双击打开 index.html（可能会有跨域限制，可用方式A避免）

前端默认请求后端地址为 http://localhost:8000（见 frontend/app.js 中 API_BASE）。如需修改端口或部署到其他机器，请调整该常量。

## 接口说明
- POST /api/infer/upload
  - form-data: file (application/json)
  - resp: { session_id, probability: float[0,1], recommended: [{measure, reason}], similar_cases: [{measure, frequency}] }

- POST /api/infer/augment
  - json: { session_id: string, text: string }
  - resp: 同上

后端使用内存保存 session，若进程重启会丢失，适用于展示场景。

## 样例 JSON 格式（可参考）
```
{
  "history": { "MI": true },
  "vitals": { "MAP": 58, "HR": 118, "CI": 2.0, "PAWP": 22 },
  "labs": { "lactate": 3.2, "EF": 30, "BNP": 1200, "urine_output_6h": 0.3 }
}
```

## 说明
- 本演示仅展示交互与可视化流程，非真实医疗决策支持系统。
- 可根据真实模型替换后端推理逻辑，保持接口契约不变即可。