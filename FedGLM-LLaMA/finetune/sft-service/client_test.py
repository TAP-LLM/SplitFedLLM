from flask import Flask, request
import torch
import io

app = Flask(__name__)

@app.route('/receive_model', methods=['POST'])
def receive_model():
    tensor_b = io.BytesIO(request.data)
    tensor = torch.load(tensor_b)
    print(tensor)
    return 'received!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
