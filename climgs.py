# clear unused images in the markdown file
import os
import re

img_dir = './images/posts/2025-03-29-Diffusion_Model'
md_path = './_posts/2025-03-29-Diffusion_Model.md'

# Get the content of the markdown file
with open(md_path, 'r', encoding='utf-8') as f:
    content = f.read()

# regex to find all referenced images in the markdown file
referenced = set(re.findall(r'/images/posts/[\w\-\.]+/([\w\-\.]+\.(?:png|jpg|jpeg|gif|bmp|svg))', content, re.IGNORECASE))

# 列出文件夹里所有实际图片  
unused = []  
for file in os.listdir(img_dir):  
    if file.lower().endswith(('.png', '.jpg', '.gif', '.jpeg', '.bmp', '.svg')):  
        if file not in referenced:  
            unused.append(file)  

print(f'未被引用图片共: {len(unused)} 张')  
for file in unused:  
    print('删除:', file)  
    os.remove(os.path.join(img_dir, file))  

print('done!')
