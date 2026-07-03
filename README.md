# Made for SECourses Premium Members : https://www.patreon.com/posts/162527725
## Download Link : https://www.patreon.com/posts/162527725

- With Ideogram 4 model as you know JSON prompting is now a thing and we need JSON prompts for both inference and training    
- Therefore, a new app was necessary to solve this issue and I built the very best local app out there for this task    
- Full features of the app introduced below with screenshots so please read
   - The installer will auto download all the necessary models with 16 connections + SHA256 hash verification
- Hopefully a tutorial video coming as well    
- SwarmUI and ComfyUI zip files updated for Ideogram 4 model, model downloads and presets and workflows already added    
   - SwarmUI : [https://www.patreon.com/SECourses/posts/114517862](https://www.patreon.com/SECourses/posts/114517862)        
   - ComfyUI : [https://www.patreon.com/SECourses/posts/105023709](https://www.patreon.com/SECourses/posts/105023709)
        
- **Windows Requirements**    
   - Python 3.11.x, FFmpeg, CUDA 13, cuDNN 9.17 or above, C++ tools, MSVC and Git        
      - Don't worry CUDA 13 works with all GPUs - make sure you have updated NVIDIA driver            
      - Follow this requirements tutorial video exactly : [https://youtu.be/DrhUHnYfwC0](https://youtu.be/DrhUHnYfwC0)            
      - Follow its updated post with links and screenshots exactly : [https://www.patreon.com/SECourses/posts/111553210](https://www.patreon.com/SECourses/posts/111553210)
            
- **For RunPod, SimplePod, Massed Compute and Linux please follow:**    
   - Massed\_Compute\_Instructions\_READ.txt        
   -  Runpod\_SimplePod\_Ultimate\_Caption\_Instructions.txt        
- **The application runs on Torch 2.11 with CUDA 13, supports literally every GPU out there including server GPUs**    
   - Moreover, we are using latest libraries such as Triton 3.7.1, Transformers 5.12.1, Bitsandbytes 0.49.2 thus we have the ultimate speed
 
## 3 July 2026 V1.2

- Qwen image captioning made more robust 
   - Such as in some cases it was adding imgur links and not anymore this bug exists
- Apply Box Edits button is now Apply Box Edits & Save so every edit is automatically saved in the outputs folder
   - Overwrites generated json file and re-generates boxed image
- The changes you made in JSON Box Preview or JSON Elements were not being saved in outputs folder and now they will be saved
- JSON Prompt Builder significantly improved
   - Now it will accurately recognize selected file's accurate outputs folder path and all changes will be saved
   - If your file is not in outputs folder, use Browse File to load file
   - Now lets say you started empty design and saved, it will be saved in a new folder inside outputs folder and keep using that folder as long as you work on that json prompt
- When you were switching between folders, it was not properly updating displayed values and this issue fixed
- Now when you switch Saved Outputs tab it will auto refresh and show latest
   - Now Saved Outputs tab is auto sorted by latest but you can re-sort by clicking display headers
- To update just run Windows_Install_Update_App.bat file
   - Zip file is still same
 
<img width="1021" height="851" alt="image" src="https://github.com/user-attachments/assets/b6bbb10d-7e97-41e1-9953-6126dc4b1537" />


## Ultimate Image Captioner Pro Features

**_Click on images to see them full resolution_**

- 1-Click to install on Windows, RunPod, Massed Compute (Linux users please use Massed Compute scripts) and SimplePod

<img height="600" alt="image" src="https://github.com/user-attachments/assets/d1806e29-e0d0-4144-9481-588f12b4030f" />

- Fully support JSON prompt generation

<img height="600" alt="image" src="https://github.com/user-attachments/assets/5fbb460c-347c-4b10-a3f3-e2d8f8e4714c" />

- Full custom user preset save and load, after restart remembers last saved / used one

<img width="1818" height="402" alt="image" src="https://github.com/user-attachments/assets/7695abf5-c806-4dc2-8293-031fe8e431bf" />

- Fully edit generated json values content or boxes and reconstruct json and easy 1-click to copy

  - Hide / display boxes to easy work, fully drag to change position or resize and make bigger or smaller from interface
 
<img  height="600" alt="image" src="https://github.com/user-attachments/assets/dd901498-340c-4a54-991e-62fd125eb65f" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/dd133c16-220a-403f-a20e-3243e0054bde" />

- We have got 35 Ideogram 4 presets ready to select and use

<img height="600" alt="8" src="https://github.com/user-attachments/assets/3289da05-b6c3-48b9-8516-4a4874aab37f" />

- Fully working automatically selected GPU VRAM presets for both Qwen and Joy Caption models

<img width="506" height="628" alt="image" src="https://github.com/user-attachments/assets/5cb943d2-ef66-4bd3-be30-3a17d96cd64b" />

- You can run as subprocess thus it will leave 0 VRAM or RAM usage after captioning
- Moreover, you can set which GPU ID to run captioning on, or set multiple GPUs to distribute batch captioning

<img width="1071" height="545" alt="image" src="https://github.com/user-attachments/assets/7298cc15-e187-4f3b-b3cd-9c3c29398db7" />

- Auto save box drawn images
- Add suffix, prefix or word replaces to generated captions automatically

<img height="600" alt="image" src="https://github.com/user-attachments/assets/30fdb0b4-2dae-4f8d-9d25-b892b65a1dee" />

<img height="600" alt="11" src="https://github.com/user-attachments/assets/b0552758-aa47-49cb-922d-7c54d810a299" />

- All Joy Caption Models are supported with full features like Qwen (e.g. batch captioning, VRAM presets, various preset prompts and save options, etc.)

<img height="600" alt="2" src="https://github.com/user-attachments/assets/a5b2dde0-eab6-465b-abb1-a947c8fe41a8" />

- Fully working JSON prompt builder that you can build from scratch or load existing image
  - Add boxes, write info, modify boxes, move them, resize them, etc. all fully working
- If you load existing image, if that image has JSON file, it will be auto loaded
  - This way, you can load your previous generations and modify and work further on them
- Every generated output is saved in outputs folder with full metadata as well

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/7e47b582-b2e2-4a02-ab8f-622e8b45c453" />

<img height="600" alt="image" src="https://github.com/user-attachments/assets/f62e732c-6716-4a32-aa7b-6eb20b8312d1" />

- Fully working view saved outputs screen that you can quickly navigate and find your previous generations and see them with full details, info, etc.
  - The page has full filtering and pagination features so use them as well

<img height="600" alt="image" src="https://github.com/user-attachments/assets/9bc8b42f-faa5-49bd-84b6-666c3bf7764e" />

- CMD screen shows full details of what is happening even token / second as well
  - Following CMD outputs is especially useful for batch folder processing
 
<img height="600" alt="7" src="https://github.com/user-attachments/assets/f8d6bb39-1f52-4273-ae38-6c8732055661" />

- Qwen JSON prompt generator is so amazing that it has a specific field for written text on images, analyze below image to understand logic
- Pay attention to JSON Elements table you will see text
- You can fully edit the JSON Elements table and re-generate, auto generated JSON as you wish
  - When you click Apply Box Edits & Save it will overwrite generated JSON file and boxes drawn image in respected output folder
 
<img height="600" alt="6" src="https://github.com/user-attachments/assets/6bfcc611-a5c7-4284-9584-e7664f079398" />

- Saved output json files are beautified and saved - can be disabled if you wish

<img height="600" alt="image" src="https://github.com/user-attachments/assets/4ec1706f-8a82-41b9-a45b-1ef96d55f7e0" />

