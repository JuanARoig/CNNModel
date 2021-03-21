namespace neuralnet
{
    public partial class ConvolutionalNN
    {
        private double[] GeneralCNN(double[,,] inputData) 
        {
            int elementAmount = inputData.GetLength(0)*inputData.GetLength(1)*inputData.GetLength(2);
            int kernelListCounter = 0; 
            double[,,] current = new double[0,0,0];
            while(elementAmount < this.transitionElementAmount && kernelListCounter < convLayerKernels.Capacity)
            {
                //reassign elementAmount after convolution and pooling
                int kernelIndex = 0;
                int convLayerIndex = 0;
                this.convLayerNetwork.Add(inputData);
                current = convLayerNetwork[convLayerIndex];//first time

                while ((current.GetLength(0)*current.GetLength(1)*current.GetLength(2) < this.transitionElementAmount) 
                && kernelIndex < convLayerKernels.Capacity)
                {
                    double[,,] currentKernel = convLayerKernels[kernelIndex];
                    current = convLayerNetwork[convLayerIndex];
                    int[] currentKernelDimensions = new int[] { currentKernel.GetLength(0), 
                                                                      currentKernel.GetLength(1),
                                                                      currentKernel.GetLength(2) };
                    this.convLayerNetwork.Add(REluOperation(MaxPoolingOperation(ConvolutionOperation(current, currentKernel),
                                                                                       currentKernelDimensions)));
                }
            }
            //converting 3d tensor to 1d vector
            double[] transitionVector = new double[current.GetLength(0)*current.GetLength(1)*current.GetLength(2)];
            double[,,] transitionTensor = convLayerNetwork[convLayerNetwork.Capacity - 1];
            int transitionAccum = 0;
            for (int i = 0; i < convLayerNetwork[convLayerNetwork.Capacity - 1].GetLength(0); i++)
            {
                for (int j = 0; j < convLayerNetwork[convLayerNetwork.Capacity - 1].GetLength(1); j++)
                {
                    for (int k = 0; k < convLayerNetwork[convLayerNetwork.Capacity - 1].GetLength(2); k++)
                    {
                        transitionVector[transitionAccum] = transitionTensor[i, j, k];
                        transitionAccum++;
                    }
                }
            }
            double[] currentLayer = new double[0];
            for (int i = 0; i < this.weights.Capacity; i++)
            {
                if (i == 0)
                {
                    currentLayer = this.Add(SoftmaxOperation(MatrixVectorProduct(weights[i], transitionVector)), biases[i]);
                }
                else
                {
                    currentLayer = this.Add(SoftmaxOperation(MatrixVectorProduct(weights[i], currentLayer)), biases[i]);
                }
            }
            return currentLayer;
        }
    }
}