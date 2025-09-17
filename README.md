# MarKollage

### Description

MarKollage is an photo collage generation program that uses computer vision and Markov models to automatically generate a visually coherent sequence of cropped images from a single photo. The program selects and arranges crops, making it easy for users to create unique collages and to explore different perspectives and compositions in their own images. The crops are selected based on composition quality which calculated using various measures of photographic principles and concepts. Once the crops have been generated, they are used to create a Markov transition matrix which pieces together the final output sequence.

### Set Up

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your images to the `assets/` folder**

3. **Run MarKollage:**
   ```bash
   # Process a single image
   python markov_photo_cropper.py assets/your_image.jpg
   
   # Process all images in assets/ folder
   python markov_photo_cropper.py --batch
   
   # Customize sequence length and output directory
   python markov_photo_cropper.py assets/your_image.jpg -l 6 -o my_results
   ```

4. **Results:** Check the output folder for `sequence_collage.png` and `sequence_visualization.png`

### Describe how the system is personally meaningful to you (at least 1 paragraph).

Photography is a hobby that I've picked up in the last couple of years, 
A large part of why I enjoy photography is because of how personal and reflective it is of the person taking the photo. There's no single or right perspective and shot to take of a subject, landscape, etc and so 
### Explain how working on it genuinely challenged you as a computer scientist (at least 1 paragraph).

    How did you push yourself outside of your comfort zone?

    My first iterations of the program used YOLO picture detection model to achieve cropped shots of subjects (people, objects, etc). Although this approach ended up causing issues down the road and I chose to pivot, this experimentation was definitely outside my comfort zone working with a type of model I'd never touched before. Although there were many bumps along the road using Github, this project has left me with a better understanding of how to properly use version control with my projects. Using git in practice is a lot different than knowing how git works, and so this was a valuable learning challenge in that sense that will definitely benefit how I approach future projects. The actual process running this program isn't ideal and I think it would definitely benefit from some sort of user interface to make it more accessible. Going forward, if time permits, I'd like to develop interfaces for programs/tools so it's easier to share and use them.

### Include a discussion of whether you believe your system is creative (and why or why not).

### Sources
Cursor
https://www.geeksforgeeks.org/python/opencv-python-tutorial/
