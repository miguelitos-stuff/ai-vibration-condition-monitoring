def conv_output_shape(input_shape, kernel_size, stride, padding):
    # Unpack the input shape
    batch_size, in_channels, height, width = input_shape

    # Calculate the output height and width
    out_height = (height + 2*padding - kernel_size) // stride + 1
    out_width = (width + 2*padding - kernel_size) // stride + 1
    return (batch_size, filter_count, out_height, out_width)

# Input shape: (batch_size, in_channels, height, width)
input_shape = (1, 10, 224, 224)

# Convolutional layer parameters
filter_count = 60
kernel_size = 5
stride = 2
padding = 0

# Calculate output shape
output_shape = conv_output_shape(input_shape, kernel_size, stride, padding)

print(output_shape)
