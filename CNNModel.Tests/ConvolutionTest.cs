using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;
using neuralnet;

namespace CNNModel.Tests
{
    public class CNNTests : ConvolutionalNN
    {

    	[Theory]
    	public void ConvolutionTest(double[,,] testInputTensor, double[,,] testKernel)
    	{
		
    	}
		[Theory]
		public void AvgPoolingTest(double[,,] testInputTensor, double[] testPrevKernelDimensions)
		{

		}
		[Theory]
		public void MaxPoolingTest(double[,,] testInputTensor, double[] testPrevKernelDimensions)
		{

		}
		public static IEnumerable<double[,,]> GetTestInput()
		{
			yield return new double[,,] {{{2, 2}}};
			yield return new double[,,] {{{2},{6},{6}},{{6}, {6}, {6}}}
		}
		[Theory]
		[ClassData(GetTestInput())]
		public void RElu_Evaluates(double[,,] testInput)
		{
			Assert.True(REluOperation(testInput) != null);
		}
		[Fact]
		public void RElu_Throws_Exception()
		{
			Assert.Throws<Exception>(() => REluOperation(new double[,,] {{{}}}));
		}
		[Theory]
		public void SoftmaxTest(double[] testInput)
		{
			
		}
		[Theory]
		public void MatrixVectorProductTest(double[] testInput)
		{
			
		}
		[Theory]
		public void AddTest(double[] testInput)
		{
			
		}
		[Theory]
		public void GeneralCNN(double[,,] testInput)
		{

		}
    }
}
