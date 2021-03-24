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
    	[Fact]
    	public void ConvolutionTest()
    	{
		
    	}
		[Fact]
		public void AvgPoolingTest()
		{

		}
		[Fact]
		public void MaxPoolingTest()
		{

		}
		[Fact]
		public void RElu_Evaluates_Input1()
		{
			Assert.True(REluOperation(new double[,,] {{{2, 2}}}) != null);
		}
		[Fact]
		public void RElu_Evaluates_Input2()
		{
			Assert.True(REluOperation(new double[,,] {{{2},{6},{6}},{{6}, {6}, {6}}}) != null);
		}
		[Fact]
		public void RElu_Throws_Exception()
		{
			Assert.Throws<Exception>(() => REluOperation(new double[,,] {{{}}}));
		}
		[Fact]
		public void SoftmaxTest()
		{
			
		}
		[Fact]
		public void MatrixVectorProductTest()
		{
			
		}
		[Fact]
		public void AddTest()
		{
			
		}
		[Fact]
		public void GeneralCNN()
		{

		}
    }
}
