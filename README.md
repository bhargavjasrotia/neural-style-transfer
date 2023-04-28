# neural-style-transfer

Neural style transfer is a technique that combines the content of one image with the style of another image. It involves using a pre-trained convolutional neural network to extract features from the input images and minimizing a loss function that measures the difference between the generated image and the target style and content images. The result is a new image that has the same content as the content image, but with the style of the style image. Neural style transfer has applications in fields such as art, computer vision, and graphics, and has gained popularity for its ability to generate visually appealing images.


# Architecture

neural style transfer uses a pretrained convolution neural network. Then to define a loss function which blends two images seamlessly to create visually appealing art, NST defines the following inputs:

A content image (c) — the image we want to transfer a style to
A style image (s) — the image we want to transfer the style from
An input (generated) image (g) — the image that contains the final result (the only trainable variable)



# loss function

content loss --->
<img width="398" alt="image" src="https://user-images.githubusercontent.com/82800949/235242629-119f5e30-3f8a-40e0-af30-d488f1ab6289.png">

style loss  --->
<img width="327" alt="image" src="https://user-images.githubusercontent.com/82800949/235242703-c84f4593-681f-48ac-b111-c513088063b6.png">


total loss --->
<img width="372" alt="image" src="https://user-images.githubusercontent.com/82800949/235242763-bebd97fc-4d4f-4380-bfc5-f0d7c60447cb.png">

here, alpha is content_weight and beta is style_weight


