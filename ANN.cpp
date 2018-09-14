//gcc -g -Wall -o matrix matrix.c

#include<iostream>
#include<time.h>
#include<math.h>
#include<stdlib.h>
#include<fstream>
#include<omp.h>

#define DBG 0
#define THREAD_COUNT 4

using namespace std;

__global__ void train(int *n_layers, int *layer, int *input_train, int *output_train, double *b, double *w)
{
	double **local_a = new double*[n_layers-1];
	for(int i=1;i<n_layers;i++)
		local_a[i] = new double[layer[i]];

	//forward propagation with TRAIN DATA
	local_a[0]=input_train[instance];
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i+1];j++)
		{
			z=b[i][j];
			for(int k=0;k<layer[i];k++)
				z+=local_a[i][k]*w[i][k][j];
			local_a[i+1][j]=1.0/(1.0+exp(-z));
		}

	//calculate error_train
	double instance_error=0.0;
	for(int i=0;i<layer[n_layers-1];i++)
		instance_error+=0.5*pow(output_train[instance]-local_a[instance],2);
	error_train+=instance_error;

	//backward propagation
	//for the last layer
	int m=n_layers-2;

	for(int i=0;i<layer[m+1];i++)
	{
		d_b[m][i]+=(local_a[m+1][i]-output_train[instance][i])*(local_a[m+1][i]*(1.0-local_a[m+1][i]));
		for(int j=0;j<layer[m];j++)
			d_w[m][j][i]+=d_b[m][i]*local_a[m][j];
	}

	//for another layers
	for(int m=n_layers-3;m>-1;m--)
		for(int i=0;i<layer[m+1];i++)
		{
			z=0.0;
			for(int j=0;j<layer[m+2];j++)
				z+=w[m+1][i][j]*d_b[m+1][j];
			d_b[m][i]+=z*(local_a[m+1][i]*(1.0-local_a[m+1][i]));
			for(int j=0;j<layer[m];j++)
				d_w[m][j][i]+=d_b[m][i]*local_a[m][j];
		}
}

double mse(double *t,double *a,int n)
{
	double sum=0.0;
	for(int i=0;i<n;i++)
		sum+=0.5*pow(t[i]-a[i],2);
	return sum;
}

int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1=i&255;
	ch2=(i>>8)&255;
	ch3=(i>>16)&255;
	ch4=(i>>24)&255;
	return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

int main(int argc,char* argv[])
{

	if(argc==1) cout<<argv[0]<<" n_layers [layers] n_epochs alpha";

	cout.precision(8);

	//internal variables
	int n_images_train;
	int n_images_test;
	double error_train,error_test,instance_error;

	//variable for continuos space asignation
	int spaces,spaces1;

	//read binary files variable
	int magic_number;
	unsigned char temp;

	cout<<" Artificial Neural Network with n-layers";
	cout<<endl<<"========================================="<<endl<<endl;

	//read args
	int n_layers=strtol(argv[1],NULL,10);
	if(DBG) cout<<endl<<"n_layers = "<<n_layers;

	int *layer = new int[n_layers];
	for(int i=0;i<n_layers;i++)
	{
		layer[i]=strtol(argv[i+2],NULL,10);
		if(DBG) cout<<endl<<"layer["<<i<<"] = "<<layer[i];
	}

	int n_epoch=strtol(argv[n_layers+2],NULL,10);
	if(DBG) cout<<endl<<"n_epoch = "<<n_epoch;

	double alpha=strtod(argv[n_layers+3],NULL);
	if(DBG) cout<<endl<<"alpha = "<<alpha;

	//read MNIST input_train
	double **input_train;

	ifstream file_in_train("../MNIST/train-images.idx3-ubyte",ios::binary);
	if(file_in_train.is_open())
	{
		int n_rows=0;
		int n_cols=0;

		file_in_train.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input_train = "<<magic_number;

		file_in_train.read((char*)&n_images_train,sizeof(n_images_train));
		n_images_train=ReverseInt(n_images_train);
		if(DBG) cout<<endl<<"n_images_train in input_train = "<<n_images_train;

		//continuous space modification		
		input_train = new double*[n_images_train];		
		input_train[0] = new double[n_images_train*layer[0]];
		for(int i=1;i<n_images_train;++i)
			input_train[i]=input_train[i-1]+layer[0];

		//old space asignation		
		//input_train = new double*[n_images_train];
		//for(int i=0;i<n_images_train;i++)
		//	input_train[i] = new double[layer[0]];

		file_in_train.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		if(DBG) cout<<endl<<"n_rows in input_train = "<<n_rows;

		file_in_train.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		if(DBG) cout<<endl<<"n_cols in input_train = "<<n_cols;

		for(int i=0;i<n_images_train;i++)
			for(int r=0;r<n_rows;r++)
				for(int c=0;c<n_cols;c++)
				{
					file_in_train.read((char*)&temp,sizeof(temp));
					input_train[i][(n_rows*r)+c]=(double)temp/255.0;
				}
		file_in_train.close();
		printf("\nSuccess Reading input_train Data");
	}
	else
	{
		cout<<endl<<"Can not read input_train data ... !!!"<<endl;
		throw;
	}

	//read MNIST output_train
	double **output_train;
	ifstream file_out_train("../MNIST/train-labels.idx1-ubyte",ios::binary);
	if (file_out_train.is_open())
	{

		file_out_train.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in output_train = "<<magic_number;

		file_out_train.read((char*)&n_images_train,sizeof(n_images_train));
		n_images_train=ReverseInt(n_images_train);
		if(DBG) cout<<endl<<"n_images_train in output_train = "<<n_images_train;

		//continuous space modification		
		output_train = new double*[n_images_train];		
		output_train[0] = new double[n_images_train*layer[n_layers-1]];
		for(int i=1;i<n_images_train;++i)
			output_train[i]=output_train[i-1]+layer[n_layers-1];

		//old space asignation
		//output_train = new double*[n_images_train];
		//for(int i=0;i<n_images_train;i++)
		//	output_train[i] = new double[layer[n_layers-1]];

		for(int i=0;i<n_images_train;i++)
			for(int j=0;j<layer[n_layers-1];j++)
				output_train[i][j]=0;

		for(int i=0;i<n_images_train;i++)
		{
			file_out_train.read((char*)&temp,sizeof(temp));
			output_train[i][(int)temp]=1;
		}
		file_out_train.close();
		printf("\nSuccess Reading output_train Data");
	}
	else
	{
		cout<<endl<<"Can not read output_train data ... !!!"<<endl;
		throw;
	}

	//read MNIST input_test
	double **input_test;

	ifstream file_in_test("../MNIST/t10k-images.idx3-ubyte",ios::binary);
	if(file_in_test.is_open())
	{
		int n_rows=0;
		int n_cols=0;

		file_in_test.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in input_test = "<<magic_number;

		file_in_test.read((char*)&n_images_test,sizeof(n_images_test));
		n_images_test=ReverseInt(n_images_test);
		if(DBG) cout<<endl<<"n_images_test in input_test = "<<n_images_test;

		//continuous space modification		
		input_test = new double*[n_images_test];		
		input_test[0] = new double[n_images_test*layer[0]];
		for(int i=1;i<n_images_test;++i)
			input_test[i]=input_test[i-1]+layer[0];

		//old space asignation
		//input_test = new double*[n_images_test];
		//for(int i=0;i<n_images_test;i++)
		//	input_test[i] = new double[layer[0]];

		file_in_test.read((char*)&n_rows,sizeof(n_rows));
		n_rows= ReverseInt(n_rows);
		if(DBG) cout<<endl<<"n_rows in input_test = "<<n_rows;

		file_in_test.read((char*)&n_cols,sizeof(n_cols));
		n_cols= ReverseInt(n_cols);
		if(DBG) cout<<endl<<"n_cols in input_test = "<<n_cols;

		for(int i=0;i<n_images_test;i++)
			for(int r=0;r<n_rows;r++)
				for(int c=0;c<n_cols;c++)
				{
					file_in_test.read((char*)&temp,sizeof(temp));
					input_test[i][(n_rows*r)+c]=(double)temp/255.0;
				}
		file_in_test.close();
		printf("\nSuccess Reading input_test Data");
	}
	else
	{
		cout<<endl<<"Can not read input_test data ... !!!"<<endl;
		throw;
	}

	//read MNIST output_test
	double **output_test;
	ifstream file_out_test("../MNIST/t10k-labels.idx1-ubyte",ios::binary);
	if (file_out_test.is_open())
	{

		file_out_test.read((char*)&magic_number,sizeof(magic_number));
		magic_number=ReverseInt(magic_number);
		if(DBG) cout<<endl<<"magic_number in output_train = "<<magic_number;

		file_out_test.read((char*)&n_images_test,sizeof(n_images_test));
		n_images_test=ReverseInt(n_images_test);
		if(DBG) cout<<endl<<"n_images_test in output_train = "<<n_images_test;

		//continuous space modification		
		output_test = new double*[n_images_test];		
		output_test[0] = new double[n_images_test*layer[n_layers-1]];
		for(int i=1;i<n_images_test;++i)
			output_test[i]=output_test[i-1]+layer[n_layers-1];
	
		//old space asignation
		//output_test = new double*[n_images_test];
		//for(int i=0;i<n_images_test;i++)
		//	output_test[i] = new double[layer[n_layers-1]];
		
		for(int i=0;i<n_images_test;i++)
			for(int j=0;j<layer[n_layers-1];j++)
				output_test[i][j]=0;

		for(int i=0;i<n_images_test;i++)
		{
			file_out_test.read((char*)&temp,sizeof(temp));
			output_test[i][(int)temp]=1;
		}
		file_out_test.close();
		printf("\nSuccess Reading output_test Data");
	}
	else
	{
		cout<<endl<<"Can not read output_test data ... !!!"<<endl;
		throw;
	}

	//create structure of data for forward
	double z;
	
	//continuous space modification
	spaces=0;
	double **b = new double*[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		spaces+=layer[i+1];
	b[0] = new double[spaces];
	for(int i=1;i<n_layers-1;i++)
		b[i]=b[0]+layer[i];

	//old space asignation
	//double **b = new double*[n_layers-1];
	//for(int i=0;i<n_layers-1;i++)
	//	b[i] = new double[layer[i+1]];

	//continuous space modification
	spaces=0;
	double ***w = new double**[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		spaces+=layer[i];
	w[0] = new double*[spaces];
	for(int i=1;i<n_layers-1;i++)
		w[i]=w[0]+layer[i-1];
	
	spaces1=0;
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i];j++)
			spaces1+=layer[i+1];
	w[0][0] = new double[spaces1];
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i];j++)
			if(i!=0 || j!=0)
				w[i][j] = w[i][j-1]+layer[i];

	//old space asignation
	//double ***w = new double**[n_layers-1];
	//for(int i=0;i<n_layers-1;i++)
	//	w[i] = new double*[layer[i]];
	//	for(int j=0;j<layer[i];j++)
	//		w[i][j] = new double[layer[i+1]];
	
	//generating data for weights
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i+1];j++)
		{
			b[i][j]=((double)rand()/(RAND_MAX))/n_images_train;
			if(DBG) cout<<endl<<"b["<<i<<"]["<<j<<"] = "<<b[i][j];
		}
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i];j++)
			for(int k=0;k<layer[i+1];k++)
			{
				w[i][j][k]=((double)rand()/(RAND_MAX))/n_images_train;
				if(DBG) cout<<endl<<"w["<<i<<"]["<<j<<"]["<<k<<"] = "<<w[i][j][k];
			}
	
	//create structure of data for backward

	//continuous space modification
	spaces=0;
	double **d_b = new double*[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		spaces+=layer[i+1];
	d_b[0] = new double[spaces];
	for(int i=1;i<n_layers-1;i++)
		d_b[i]=d_b[0]+layer[i];

	//old space asignation
	//double **d_b = new double*[n_layers-1];
	//for(int i=0;i<n_layers-1;i++)
	//	d_b[i] = new double[layer[i+1]];
	
	//continuous space modification
	spaces=0;
	double ***d_w = new double**[n_layers-1];
	for(int i=0;i<n_layers-1;i++)
		spaces+=layer[i];
	d_w[0] = new double*[spaces];
	for(int i=1;i<n_layers-1;i++)
		d_w[i]=d_w[0]+layer[i-1];
	
	spaces1=0;
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i];j++)
			spaces1+=layer[i+1];
	d_w[0][0] = new double[spaces1];
	for(int i=0;i<n_layers-1;i++)
		for(int j=0;j<layer[i];j++)
			if(i!=0 || j!=0)
				d_w[i][j] = d_w[i][j-1]+layer[i];


	//old space asignation
	//double ***d_w = new double**[n_layers-1];
	//for(int i=0;i<n_layers-1;i++)
	//{
	//	d_w[i] = new double*[layer[i]];
	//	for(int j=0;j<layer[i];j++)
	//		d_w[i][j] = new double[layer[i+1]];
	//}

	//execution in epochs

	cudaMalloc( (void**)&dev_,N*sizeof(int)));
	cudaMemcpy( dev_,,N*sizeof(int),cudaMemcpyHostToDevice));
	add<<<128,128>>>(dev_);

	for(int epoch=0;epoch<n_epoch;epoch++)
	{
		error_train=0.0;
		//for every instance in train

		//init to 0 all deltas
		for(int i=0;i<n_layers-1;i++)
			for(int j=0;j<layer[i+1];j++)
				d_b[i][j]=0;

		for(int i=0;i<n_layers-1;i++)
			for(int j=0;j<layer[i];j++)
				for(int k=0;k<layer[i+1];k++)
					d_w[i][j][k]=0;

		for(int instance=0;instance<n_images_train;instance++)
		{

			double **local_a = new double*[n_layers-1];
			for(int i=1;i<n_layers;i++)
				local_a[i] = new double[layer[i]];

			//forward propagation with TRAIN DATA
			local_a[0]=input_train[instance];
			for(int i=0;i<n_layers-1;i++)
				for(int j=0;j<layer[i+1];j++)
				{
					z=b[i][j];
					for(int k=0;k<layer[i];k++)
						z+=local_a[i][k]*w[i][k][j];
					local_a[i+1][j]=act(z);
				}

			//calculate error_train
			instance_error=mse(output_train[instance],local_a[n_layers-1],layer[n_layers-1]);
			error_train+=instance_error;

			//backward propagation
			//for the last layer
			int m=n_layers-2;

			for(int i=0;i<layer[m+1];i++)
			{
				d_b[m][i]+=derror(output_train[instance][i],local_a[m+1][i])*dact(local_a[m+1][i]);
				for(int j=0;j<layer[m];j++)
					d_w[m][j][i]+=d_b[m][i]*local_a[m][j];
			}

			//for another layers
			for(int m=n_layers-3;m>-1;m--)
				for(int i=0;i<layer[m+1];i++)
				{
					z=0.0;
					for(int j=0;j<layer[m+2];j++)
						z+=w[m+1][i][j]*d_b[m+1][j];
					d_b[m][i]+=z*dact(local_a[m+1][i]);
					for(int j=0;j<layer[m];j++)
						d_w[m][j][i]+=d_b[m][i]*local_a[m][j];
				}
		}

		error_test=0.0;
		//for every instance in test
		for(int instance=0;instance<n_images_test;instance++)
		{

			double **local_a = new double*[n_layers-1];
			for(int i=1;i<n_layers;i++)
				local_a[i] = new double[layer[i]];

			//forward propagation with TEST DATA
			local_a[0]=input_test[instance];
			for(int i=0;i<n_layers-1;i++)
				for(int j=0;j<layer[i+1];j++)
				{
					z=b[i][j];
					for(int k=0;k<layer[i];k++)
						z+=local_a[i][k]*w[i][k][j];
					local_a[i+1][j]=act(z);
				}

			//calculate error_test
			instance_error=mse(output_test[instance],local_a[n_layers-1],layer[n_layers-1]);
			error_test+=instance_error;
		}

		cout<<endl<<"Error en epoca "<<epoch<<","<<error_train/n_images_train<<","<<error_test/n_images_test;

		//new weight and bias
		for(int m=0;m<n_layers-1;m++)
			for(int j=0;j<layer[m+1];j++)
			{
				b[m][j]-=alpha*d_b[m][j];
				for(int i=0;i<layer[m];i++)
					w[m][i][j]-=alpha*d_w[m][i][j];
			}

	}

	return 0;
}

