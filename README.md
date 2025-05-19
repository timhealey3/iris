# iris
Learning how Pytorch works, using the Iris data set

## Model
Model is a simple Neural Network. It implements matrix multiplication and bias addition manually, instead of using torch.matmul and torch.add. This is to learn how these operations work.

## Data
Data is loaded using torch.utils.data.Dataset and torch.utils.data.DataLoader. The data is split into training and testing sets using torch.utils.data.random_split. The data is also shuffled using torch.Generator.

## Training
Training is done using torch.optim.Adam. The loss is calculated using torch.nn.CrossEntropyLoss as there are 3 possible outputs.

## Linear Model
Input: 4 features (sepal length, sepal width, petal length, petal width)

Data that is passed in the model, we do matrix multiplication with the weights and add the bias.

When training, we calculate the loss, use backpropagation to compute gradients, and use the optimizer to update the weights and bias based on these gradients.

Output: 3 possible classes (setosa, versicolor, virginica)

## Non Linear Model
Three hidden layers with 16, 8, and 3 neurons. Uses ReLU activation function.