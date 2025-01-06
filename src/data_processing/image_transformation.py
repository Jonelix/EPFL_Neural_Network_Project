#IMPORTS
from PIL import Image, ImageOps, ImageDraw, ImageEnhance
import random
import numpy as np
from enum import Enum
from src.data_processing import image_transformation as it

class Transformation(Enum):
    """
    Enum class that defines a set of image transformation operations. Each transformation is represented
    by a name and a corresponding function that applies the transformation to an input image.

    Transformations include:
    - SPLIT: Splits the image into smaller grid pieces.
    - ROTATE: Rotates the image by a random degree.
    - MIRROR: Mirrors the image horizontally.
    - SUB: Extracts a random subimage from the original image.
    - SHUFFLE: Shuffles the pixels of the image.
    - CIRCLES: Adds random circles to the image.
    - BRIGHTNESS: Adjusts the brightness of the image.
    - INVERT: Inverts the colors of the image.
    - SHIFT_COLOR: Shifts the color tones of the image randomly.
    - NOISE: Adds random noise to the image.

    Attributes
    ----------
    value : tuple
        Each transformation is represented by a tuple where the first element is a string identifier
        for the transformation and the second element is the function that implements the transformation.
    """
    SPLIT = ("_spli", lambda img_in, n_pieces: split_image_into_grid(img_in, n_pieces))
    
    ROTATE = ("_rot", lambda img_in, seed: rotate_image(img_in, seed))
    MIRROR = ("_mir", lambda img_in, seed: mirror_image(img_in))
    SUB = ("_sub", lambda img_in, seed: random_subimage(img_in, seed))
    
    SHUFFLE = ("_shu", lambda img_in, seed: shuffle_image(img_in, seed))
    CIRCLES = ("_cir", lambda img_in, seed: add_random_circles(img_in, seed))
    BRIGHTNESS = ("_bri", lambda img_in, seed: set_brightness(img_in, seed))
    
    INVERT = ("_inv", lambda img_in, seed: invert_colors(img_in))
    SHIFT_COLOR = ("_shi", lambda img_in, seed: shift_color_towards_random(img_in, seed))
    NOISE = ("_noi", lambda img_in, seed: add_noise_to_image(img_in, seed))     
        
    def pipe_to_name(pipeline):
        """
        Converts a list of transformations to a string representation of their combined names.

        Parameters
        ----------
        pipeline : list
            A list of transformations to be converted into a name string.

        Returns
        -------
        str
            A concatenated string containing the names of the transformations applied in the pipeline.
        """
        name = ""
        # print(pipeline)
        if Transformation.ROTATE in pipeline:
            name += Transformation.ROTATE.value[0]
        if Transformation.MIRROR in pipeline:
            name += Transformation.MIRROR.value[0]
        if Transformation.SUB in pipeline:
            name += Transformation.SUB.value[0]
        if Transformation.SHUFFLE in pipeline:
            name += Transformation.SHUFFLE.value[0]
        if Transformation.CIRCLES in pipeline:
            name += Transformation.CIRCLES.value[0]
        if Transformation.BRIGHTNESS in pipeline:
            name += Transformation.BRIGHTNESS.value[0]
        if Transformation.INVERT in pipeline:
            name += Transformation.INVERT.value[0]
        if Transformation.SHIFT_COLOR in pipeline:
            name += Transformation.SHIFT_COLOR.value[0]
        if Transformation.NOISE in pipeline:
            name += Transformation.NOISE.value[0]
        return name
    
    def name_to_pipe(name: str):
        """
        Converts a string representation of a combined transformation name back into a list of transformations.

        Parameters
        ----------
        name : str
            A string representing the transformations applied, where each transformation is represented by its
            identifier (e.g., "_rot" for rotation).

        Returns
        -------
        list
            A list of transformations corresponding to the input name string.
        """
        pipe = []
        name_split = name.split("_")
        name_split = ["_" + part for part in name_split if part]
        if Transformation.ROTATE.value[0] in name_split:
            pipe.append(Transformation.ROTATE)
        if Transformation.MIRROR.value[0] in name_split:
            pipe.append(Transformation.MIRROR)
        if Transformation.SUB.value[0] in name_split:
            pipe.append(Transformation.SUB)
        if Transformation.SHUFFLE.value[0] in name_split:
            pipe.append(Transformation.SHUFFL)
        if Transformation.CIRCLES.value[0] in name_split:
            pipe.append(Transformation.CIRCLES)
        if Transformation.BRIGHTNESS.value[0] in name_split:
            pipe.append(Transformation.BRIGHTNESS)
        if Transformation.INVERT.value[0] in name_split:
            pipe.append(Transformation.INVERT)
        if Transformation.SHIFT_COLOR.value[0] in name_split:
            pipe.append(Transformation.SHIFT_COLOR)
        if Transformation.NOISE.value[0] in name_split:
            pipe.append(Transformation.NOISE)
        return pipe

class SamplePoint:
    """ 
    A class that represents a single sample point consisting of an input image and its corresponding ground truth.

    Parameters
    ----------
    input_image : numpy.ndarray or torch.Tensor
        The input image (e.g., a 2D or 3D image) for the sample.
    
    groundtruth : numpy.ndarray or torch.Tensor
        The ground truth corresponding to the input image, typically used for supervised learning tasks.

    Attributes
    ----------
    input_image : numpy.ndarray or torch.Tensor
        The input image associated with this sample.
    
    groundtruth : numpy.ndarray or torch.Tensor
        The ground truth associated with this sample.
    """
    def __init__(self, input_image, groundtruth):
        self.input_image = input_image
        self.groundtruth = groundtruth
     
    def get(self):
        """
        Retrieves both the input image and the ground truth as a tuple.

        Returns
        -------
        tuple
            A tuple containing the input image and the ground truth.
        """
        return self.input_image, self.groundtruth
    def get_image(self):
        """
        Retrieves only the input image.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The input image associated with this sample.
        """
        return self.input_image
    def get_groundtruth(self):
        """
        Retrieves only the ground truth.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            The ground truth associated with this sample.
        """
        return self.groundtruth

#DIVISION N x X
def split_image_into_grid(image, n):
    """
    Splits an image into an n x n grid, scales each subimage to 400x400 pixels, 
    and returns them as a list of PIL Image objects.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to be split. This should be a PIL Image object.
    
    n : int
        The number of rows and columns to split the image into, creating an n x n grid of subimages.

    Returns
    -------
    List[PIL.Image.Image]
        A list of PIL Image objects representing the subimages. Each subimage is scaled to 400x400 pixels.

    Notes
    -----
    If the dimensions of the input image are not perfectly divisible by n, 
    the subimages will still be of the computed size, but this might lead to some 
    cropping. The final size of each subimage is fixed at 400x400 pixels.

    Exmaple
    -----
    from src.data import image_transformation as it
    num = "001"
    image_in = f"./data/training/training/images/satImage_0000{num}.png"
    truth_in = f"./data/training/training/groundtruth/satImage_0000{num}.png"
    out_dir_image = f"./data/exp_image"
    out_dir_truth = f"./data/exp_truth"

    it.split_image_into_subimages(image_in, out_dir_image, 0)
    it.split_image_into_subimages(truth_in, out_dir_truth, 0)
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Calculate the dimensions of each subimage
    sub_width, sub_height = width // n, height // n

    # Generate the subimages
    subimages = []
    for row in range(n):
        for col in range(n):
            # Define the bounding box for the current subimage
            left = col * sub_width
            upper = row * sub_height
            right = left + sub_width
            lower = upper + sub_height
            box = (left, upper, right, lower)

            # Crop and scale the subimage
            sub_img = img.crop(box)
            sub_img = sub_img.resize((400, 400))
            subimages.append(sub_img)

    return subimages

#ROTATION
def rotate_image(image, seed):
    """
    Rotates an image by a specified number of degrees, crops it to the original size, 
    and fills blank space with black.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to be rotated. This should be a PIL Image object.
    
    seed : int
        The random seed used for generating the rotation angle. This ensures reproducibility of results.

    Returns
    -------
    PIL.Image.Image
        A PIL Image object representing the rotated and cropped image. The image is rotated by a 
        random number of degrees (between 1 and 360), with any resulting blank space filled with black.

    Notes
    -----
    The image is rotated on a larger canvas with a black background, and then cropped back to 
    the original size to remove any black borders that might result from the rotation.
    Example
    -----
    from src.data import image_transformation as it
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    degrees = 45                           # Rotation angle in degrees (counter-clockwise)
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.rotate_image_crop(img_in, degrees, out_dir_img, output_name="img_45_rot.png")
    it.rotate_image_crop(truth_in, degrees, out_dir_truth, output_name="truth_45_rot.png")
    """
    # Open the image
    img = image.copy()
    original_size = img.size  # Original dimensions (width, height)

    # Create a larger canvas with a black background
    larger_canvas = Image.new("RGB", img.size, (0, 0, 0))  # Black background
    larger_canvas.paste(img, (0, 0))

    # Rotate the image
    random.seed(seed)  # Set the seed for reproducibility
    degrees = random.randint(1, 360)
    rotated_img = larger_canvas.rotate(degrees, resample=Image.BICUBIC, expand=False)

    # Crop back to the original size
    rotated_cropped_img = rotated_img.crop((0, 0, original_size[0], original_size[1]))

    return rotated_cropped_img



#MIRROR (HORIZONTAL FLIP)
def mirror_image(image):
    """
    Mirrors the content of an image (flips it horizontally) and returns the result as a PIL Image object.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to be mirrored. This should be a PIL Image object.

    Returns
    -------
    PIL.Image.Image
        The mirrored image as a PIL Image object, flipped horizontally.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.mirror_image(img_in, out_dir_img, output_name="mirrored_image.png")
    it.mirror_image(truth_in, out_dir_truth, output_name="mirrored_image.png")
    """
    # Open the image
    img = image.copy()

    # Flip the image horizontally (mirror effect)
    mirrored_img = ImageOps.mirror(img)

    return mirrored_img

#STRETCH
def random_subimage(image, seed):
    """
    Selects a random subimage from the input image using the given seed.
    The subimage size is determined randomly between 10 and 400 using the seed.
    The subimage is then scaled to 400x400 pixels.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image from which a random subimage will be selected.
        
    seed : int
        The seed value for random number generation, ensuring reproducibility.

    Returns
    -------
    PIL.Image.Image
        The randomly selected and scaled subimage as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.random_subimage(img_in, 12345, out_dir_img, output_name="subimage.png")
    it.random_subimage(truth_in, 12345, out_dir_truth, output_name="subimage.png")
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Get image dimensions
    width, height = image.size

    # Generate random subimage dimensions between 10 and 400
    subimage_size = random.randint(10, 400)

    # Calculate the max allowable top-left corner coordinates
    max_x = width - subimage_size
    max_y = height - subimage_size

    # Randomly select the top-left corner of the subimage
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Define the bounding box for the subimage
    box = (x, y, x + subimage_size, y + subimage_size)

    # Crop the subimage from the original image
    subimage = image.crop(box)

    # Resize the subimage to 400x400 pixels
    subimage = subimage.resize((400, 400), Image.Resampling.LANCZOS)

    return subimage

#PUZZLE SHUFFLE
def shuffle_image(image, seed):
    """
    Divides an image into an n x n grid, shuffles the pieces randomly with a given seed, 
    and reconstructs the shuffled image.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to be shuffled.

    seed : int
        The seed for random shuffling. This ensures reproducibility if provided.

    Returns
    -------
    PIL.Image.Image
        The shuffled image as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    n = 4                           # Number of divisions (4x4 grid)
    seed = 42                       # Seed for reproducibility
    out_dir_img = "data/exp_image"   # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.shuffle_image(img_in, seed)
    it.shuffle_image(truth_in, seed)
    """
    # Open the image
    img = image.copy()
    width, height = img.size


    # Set the seed for reproducibility
    random.seed(seed)
    n =  random.randint(2, 5)

    # Calculate the size of each grid cell
    grid_width = width // n
    grid_height = height // n

    # Extract grid cells
    grid_cells = []
    for i in range(n):
        for j in range(n):
            left = j * grid_width
            upper = i * grid_height
            right = left + grid_width
            lower = upper + grid_height
            grid_cells.append(img.crop((left, upper, right, lower)))

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Shuffle the grid cells
    random.shuffle(grid_cells)

    # Create a new blank image to reconstruct the shuffled image
    shuffled_img = Image.new("RGB", (width, height))

    # Place the shuffled cells back into the new image
    for i in range(n):
        for j in range(n):
            left = j * grid_width
            upper = i * grid_height
            shuffled_img.paste(grid_cells.pop(0), (left, upper))

    return shuffled_img

#HOLES
def add_random_circles(image, seed):
    """ 
    Adds black circles to an image at random positions with random diameters.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to which circles will be added.

    seed : int
        The seed for random diameter and position generation. Ensures reproducibility.

    Returns
    -------
    PIL.Image.Image
        The modified image with added black circles as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    n = 4                           # Rotation angle in degrees (counter-clockwise)
    seed = 42                       # Seed for reproducibility
    out_dir_img = "data/exp_image"   # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.add_random_circles(img_in, 16, seed, out_dir_img, output_name="image_with_circles.png")
    it.add_random_circles(truth_in, 16, seed, out_dir_truth, output_name="image_with_circles.png")
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Maximum circle diameter
    max_diameter = min(width, height) // 8

    # Set the random seed for reproducibility
    random.seed(seed)
    num_circles =  random.randint(1, 32)

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Generate and draw circles
    for _ in range(num_circles):
        # Random diameter
        diameter = random.randint(1, max_diameter)

        # Random position ensuring the circle stays within bounds
        x = random.randint(0, width - diameter)
        y = random.randint(0, height - diameter)

        # Draw the circle
        draw.ellipse([x, y, x + diameter, y + diameter], fill="black")

    return img


#BRIGHTNESS
def set_brightness(image, seed):
    """
    Adjusts the brightness of an image based on the input brightness level.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to adjust the brightness of.

    seed : int
        The seed for random brightness level generation. Ensures reproducibility.

    Returns
    -------
    PIL.Image.Image
        The adjusted image with modified brightness as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    seed = 80                      # Random seed for brightness adjustment
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.set_brightness(img_in, seed, out_dir_img, output_name="brightness_adjusted.png")
    it.set_brightness(truth_in, seed, out_dir_truth, output_name="brightness_adjusted.png")
    """

    #Brightness
    random.seed(seed)
    brightness_level =  random.randint(0, 100)

    # Open the image
    img = image.copy()

    # Calculate brightness factor
    # 0 -> 0.0 (black), 50 -> 1.0 (no change), 100 -> 2.0 (white)
    brightness_factor = brightness_level / 50

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    adjusted_img = enhancer.enhance(brightness_factor)

    return adjusted_img


#INVERT
def invert_colors(image):
    """
    Inverts the colors of an image.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to invert the colors of.

    Returns
    -------
    PIL.Image.Image
        The image with inverted colors as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.invert_colors(img_in, out_dir_img, output_name="inverted_image.png")
    it.invert_colors(truth_in, out_dir_truth, output_name="inverted_image.png")
    """
    # Open the image
    img = image.copy()

    # Invert colors
    inverted_img = ImageOps.invert(img.convert("RGB"))

    return inverted_img


#COLORS
def shift_color_towards_random(image, seed):
    """
    Shifts the color scale of the image towards a randomly generated color based on the seed.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to shift the colors of.

    seed : int
        The seed used to generate the random color for the color shift. This ensures reproducibility.

    Returns
    -------
    PIL.Image.Image
        The color-shifted image as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    seed = 32
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.shift_color_towards_random(img_in, seed, out_dir_img, output_name="color_shifted_image.png")
    #it.shift_color_towards_random(truth_in, seed, out_dir_truth, output_name="color_shifted_image.png")
    """
    # Open the image
    img = image.copy()

    # Set the random seed for reproducibility
    random.seed(seed)

    # Generate a random color (R, G, B) where each channel is between 0 and 255
    random_color = (
        random.randint(0, 255),  # Red channel
        random.randint(0, 255),  # Green channel
        random.randint(0, 255)   # Blue channel
    )

    # Split the image into RGB channels
    r, g, b = img.split()

    # Calculate the scaling factors to shift the color balance
    r_scale = random_color[0] / 255.0
    g_scale = random_color[1] / 255.0
    b_scale = random_color[2] / 255.0

    # Apply the scaling factors to each color channel
    r = r.point(lambda i: i * r_scale)
    g = g.point(lambda i: i * g_scale)
    b = b.point(lambda i: i * b_scale)

    # Merge the modified channels back into an image
    shifted_img = Image.merge("RGB", (r, g, b))

    return shifted_img


#NOISE
def add_noise_to_image(image, seed):
    """
    Adds random noise to an image.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image to which noise will be added.

    seed : int
        The seed used to generate random noise. This ensures reproducibility.

    Returns
    -------
    PIL.Image.Image
        The noisy image as a PIL Image object.

    Examples
    --------
    img_in = "data/training/training/images/satImage_0000001.png"  # Replace with your image file
    truth_in = "data/training/training/groundtruth/satImage_0000001.png"  # Replace with your image file
    noise = 100
    out_dir_img = "data/exp_image"  # Replace with your desired output folder
    out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

    # Call the function
    it.add_noise_to_image(img_in, noise, out_dir_img, output_name="noisy_image.png")
    """
    # Open the image and convert it to a NumPy array
    img = image.copy()
    img_array = np.array(img)

    random.seed(seed)
    noise_level =  random.randint(0, 255)

    # Generate random noise
    noise = np.random.randint(-noise_level, noise_level + 1, img_array.shape, dtype=np.int16)

    # Add noise to the image and clip the values to keep them valid (0-255)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Convert the noisy image array back to a PIL Image
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img
