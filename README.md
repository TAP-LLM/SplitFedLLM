# FedGLM: Safely Learning with Private Data
https://github.com/TAP-LLM/SplitFedLLM/assets/131137567/c647c889-4c9c-43e7-b82b-6d3fbb0638b1
## Reviewer TIPS
This is a project containing FL-GLM and FL-LLaMA. If you are a reviewer for <A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability> . Please click the FL-LLaMA directory. Thank you for your support! 


## Project Introduction
This open source project is based on the open source LLAMA and GLM models, and has implemented a distributed federated learning framework for model fine-tuning and inference on a single/multiple machine deployment.
While ensuring client data privacy and security, it aggregates model parameters to achieve model parameter sharing. This allows users with limited computing power to use the resources of the project deployment platform for model fine-tuning, thereby achieving vertical domain customization of the model.

2024/7/9: Due to project funding, the ChatGLM federated learning code is currently undergoing open-source approval. We will continue to provide updates.

2024/7/16: FedGLM released！

## Supported Models
| Model            | Type | Download                                                                                                                                |
|------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------|
| ChatGLM-6B | Chat |https://github.com/THUDM/ChatGLM-6B|
| Llama-2-7b-hf    | Base | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama-2-7b-chat-hf | Chat | [🤗 Huggingface](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |



## Notices
FedGLM is currently **going through the open-source procedures**. Once the process is completed, it will be uploaded to GitHub as soon as possible. In the meantime, if you need to use FedGLM, please send an email using your academic email address (e.g., @edu.cn, @edu.com) to [sy2342110@buaa.edu.cn], stating your name, the institution you are affiliated with, and the electronic PDF version of **the Commitment Letter**. Upon review, the code will be sent to you via email. Thank you for your understanding and support.

Please note that the FedGLM folder contains training solutions based on the flower framework and ChatGLM-6B, while the FedGLM-LaMA folder contains training solutions based on the flask communication framework and LLaMA-7B, for details, please refer to the readme file in each folder. We recommend using FedGLM because the communication capability of the flower framework is more stable. In addition, FedGLM supports training with both Chinese and English, but FedGLM-LaMA is more suitable for training English.

## Commitment Letter Template in Chinese

源代码使用承诺书

我是 [您的全名]，来自 [您的学校/机构]。我现正从事 [研究领域/项目名称] 方面的研究工作。本人对你们开发的项目很感兴趣，希望获得该项目的源代码以支持我的研究活动。 在此，我郑重承诺： 
1. 我将仅在个人学习、研究或非商业用途下使用所获取的源代码。
2. 我不会将源代码直接或间接地用于任何商业产品或服务中。
3. 我不会将源代码泄露给任何第三方。
4. 若违反上述任一条款，我愿意承担由此产生的一切后果，包括但不限于法律追责。

 承诺人签名: ______________________ 日期: ________ 年 ___ 月 ___ 日
 
 联系信息: - 邮箱: [您的邮箱] - 电话: [您的电话号码]
 
 ## Commitment Letter Template in English
 
 Source Code Usage Commitment Letter
 
I, [Your Full Name], from [Your University/Institution], am currently engaged in research/work in the field of [Research Area/Project Name]. I am interested in your project and would like to obtain the source code to support my research activities.

Hereby, I solemnly commit to the following:

I will use the obtained source code solely for personal learning, research, or non-commercial purposes.
I will not use the source code directly or indirectly for any commercial products or services.
I will not disclose the source code to any third party.
If I violate any of the above terms, I am willing to bear all consequences, including but not limited to legal liabilities.
Signature: ______________________
Date: ________ Year ___ Month ___ Day

Contact Information:

Email: [Your Email]
Phone: [Your Phone Number]



## Reference
If you find this repository useful or our work is related to your research, please kindly cite it:
```
@misc{zheng2024safelylearningprivatedata,
      title={Safely Learning with Private Data: A Federated Learning Framework for Large Language Model}, 
      author={JiaYing Zheng and HaiNan Zhang and LingXiang Wang and WangJie Qiu and HongWei Zheng and ZhiMing Zheng},
      year={2024},
      eprint={2406.14898},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2406.14898}, 
}
```




