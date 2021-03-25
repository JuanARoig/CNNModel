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
    	public void Convolution_Evaluates_Input1()
    	{
			
    	}
		[Fact]
    	public void Convolution_Evaluates_Input2()
    	{
		
    	}
		[Fact]
		public void Convolution_Throws_Exception()
		{

		}
		[Fact]
		public void AvgPooling_Evaluates_Input1()
		{

		}
		[Fact]
		public void AvgPooling_Evaluates_Input2()
		{

		}
		[Fact]
		public void AvgPooling_Throws_Exception()
		{

		}
		[Fact]
		public void MaxPooling_Evaluates_Input1()
		{

		}
		[Fact]
		public void MaxPooling_Evaluates_Input2()
		{

		}
		[Fact]
		public void MAxPooling_Throws_Exception()
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
		public void SoftmaxEvaluates_Input1()
		{
			
		}
		[Fact]
		public void SoftmaxEvaluates_Input2()
		{
			
		}
		[Fact]
		public void Softmax_Throws_Exception()
		{

		}
		[Fact]
		public void MatrixVectorProduct_Evaluates_Input1()
		{
			
		}
		[Fact]
		public void MatrixVectorProduct_Evaluates_Input2()
		{
			
		}
		[Fact]
		public void MatrixVectorProduct_Throws_Exception()
		{

		}
		[Fact]
		public void Add_Evaluates_Input1()
		{
			
		}
		[Fact]
		public void Add_Evaluates_Input2()
		{
			
		}
		[Fact]
		public void Add_Throws_Exception()
		{

		}
		[Fact]
		public void GeneralCNN_Evaluates_Input1()
		{
			
		}
		[Fact]
		public void GeneralCNN_Evaluates_Input2()
		{
			
		}
		[Fact]
		public void GeneralCNN_Throws_Exception()
		{

		}
    }
}
