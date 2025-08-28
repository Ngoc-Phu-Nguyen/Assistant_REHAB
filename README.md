# REHAB HUB 
Demo a flow that can store data to clound (hub) ||
Web app that can access to data ||
Apply API AI to interact with data ||
# Technical structure
There is three main parts for this project: Backend, AI procssing and Fronted UI
<img width="667" height="420" alt="image" src="https://github.com/user-attachments/assets/b2b67529-19f6-4750-bdc7-de28c8b64a0d" />

Figure 1. Struction project 

# Backend to S3
<img width="916" height="379" alt="image" src="https://github.com/user-attachments/assets/6b5d7d95-8e28-49e6-8a98-e3236bfa34b7" />

Figure 2. From the local data, we process it, clean it and then upload to S3. 
The tool can be automate run.

# AI API
<img width="998" height="490" alt="image" src="https://github.com/user-attachments/assets/78b6eceb-f6d9-4bba-af99-6ddd5e7238bf" />

Figure 3. Data after cleaned send to API AI to process. 
The process and output depends on the prompt design

# Fronted UI
<img width="857" height="197" alt="image" src="https://github.com/user-attachments/assets/3d285be5-63fa-4d8c-a00f-2ffee71f2646" />

Figure 4. The data is transfmit thru http method to UI web
However if you want to access from outside we need a public place in internet to send these data to. 

