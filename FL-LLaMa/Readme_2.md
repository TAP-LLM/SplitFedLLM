pytorch_model_A/B/C由于是分开的模型，实际加载的时候参数名都是从layer0开始的，所以在加载的时候需要注意这一点。
pytorch_model_A的layer0实际上是模型的第0层   共0-4，五层
pytorch_model_B的layer0实际上是模型的第1层   共1-30，三十层
pytorch_model_C的layer0实际上是模型的第27层   共27-31，五层