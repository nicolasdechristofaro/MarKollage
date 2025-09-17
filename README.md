# MarKollage

### Description

MarKollage 

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
### Explain how working on it genuinely challenged you as a computer scientist (at least 1 paragraph).

    How did you push yourself outside of your comfort zone?
    Why was this an important challenge for you?
    What are the next steps for you going forward?

### Include a discussion of whether you believe your system is creative (and why or why not).

### Sources