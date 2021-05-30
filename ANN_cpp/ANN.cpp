#include<bits/stdc++.h>
#include <unistd.h>
#include <iostream>
using namespace std;
#define pb push_back
vector<vector<double> > input,validation;
vector<int> vv, digit;
const int hidden_nodes = 7;
double t[3100][10];
double soft[10];
int confusion[10][10];
int gggg, true_positive=0;
double vt = 0.0, b1 = 0.1, b2 = 0.1, eta1 = 0.05, eta2 = 0.05, epsilon = 1.0;

double sigmoid(double a)
{
   return 1.0/(1.0 + exp(-a));
}
double softmax(double A_k_output[])
{
   double temp = 0.0;
   for(int i=0; i<10; i++) temp += exp(A_k_output[i]);
   for(int i=0; i<10; i++) soft[i] = (exp(A_k_output[i]))/temp;
}
int find_max(double ak[]){
   double max = 0.0;
   int ind;
   for(int k=0; k<10; k++){
      if(ak[k]>max){
         max = ak[k];
         ind = k;
      }
   }
   return ind;
}
double del_Wkj[10][hidden_nodes], del_Wji[hidden_nodes][64];
int main()
{
   ifstream read;
   read.open("train.txt");
   string s;
   while(read>>s)
   {
      int sz = s.size();
      vector<double> temp;
      string ss;
      for(int i=0;i<s.size()-1;i++)
      {
         if(s[i]==',')
         {
            double f = atof(ss.c_str());
            ss.clear();
            temp.pb(sigmoid(f));
         }
         else
         {
            ss = ss+ s[i];
         }
      }
      int p = s[sz-1]-'0';
      digit.pb(p);
      t[gggg++][p] = 1;
      input.pb(temp);
      temp.clear();
   }
   read.close();
   read.open("validation.txt");

   while(read>>s)
   {
      int sz = s.size();
      vector<double> temp;
      string ss;
      for(int i=0;i<s.size()-1;i++)
      {
         if(s[i]==',')
         {
            double f = atof(ss.c_str());
            ss.clear();
            temp.pb(sigmoid(f));
         }
         else
         {
            ss = ss+ s[i];
         }
      }
      int p = s[sz-1]-'0';
      vv.push_back(p);
      validation.pb(temp);
      temp.clear();
   }
   read.close();
   //cout<<validation.size();
   //exit(0);

   double wji[hidden_nodes][64],wkj[10][hidden_nodes];
   srand(unsigned(time(NULL)));
   for(int j=0;j<hidden_nodes;j++)
   {
      for(int i=0; i<64; i++)
      {
         double u = (double)rand()/(RAND_MAX + 1.0);
         int f = rand()%2;
         if(f%2 == 1)
         wji[j][i] = u;
         else
         wji[j][i] = -u;
      }
   }

   for(int k=0; k<10; k++)
   {
      for(int j=0;j<hidden_nodes;j++)
      {
         double u = (double)rand()/(RAND_MAX + 1.0);
         int f = rand()%2;
         if(f%2 == 1)
         wkj[k][j] = u;
         else
         wkj[k][j] = -u;
      }
   }

   for(int epoch=0; epoch<3000; epoch++)
   {
      //cout<<epoch<<endl;
      for(int in=0; in<input.size(); in++)
      {
         double A_j_hidden[hidden_nodes],saj[hidden_nodes];
         for(int j=0; j<hidden_nodes; j++)
         {
            double temp = 0.0;
            for(int i=0; i<64; i++)
            {
               temp = temp + wji[j][i]*(input[in][i]);
            }
            A_j_hidden[j] = temp;
            saj[j] = sigmoid(temp);
         }

         //start of second layer
         double A_k_output[10];
         for(int k=0;k<10;k++)
         {
            double temp = 0.0;
            for(int j=0;j<hidden_nodes;j++)
            {
               temp = temp + wkj[k][j] * saj[j];
            }
            A_k_output[k] = temp;
         }

         softmax(A_k_output);
         double del_Wkj[10][hidden_nodes], del_Wji[hidden_nodes][64];
         for(int k=0;k<10;k++)
         {
            for(int j=0;j<hidden_nodes;j++)
            {
               del_Wkj[k][j]+= (soft[k]-t[in][k]) * saj[j];
            }
         }
         for(int i=0;i<64;i++)
         {
            for(int j=0;j<hidden_nodes;j++)
            {
               double p = 0.0;
               for(int k=0;k<10;k++)
               {
                  p =p + wkj[k][j] * ( soft[k] - t[in][k]);
               }
               del_Wji[j][i]+= p * saj[j] * (1.0 - saj[j]) * (input[in][i]);
            }
         }
         // adaptive gradient and momentum ADAM
         // updating Wkj - stochastic gradient descent

         //double change_in_wji[hidden_nodes][64],change_in_wkj[10][hidden_nodes];
         if(in%100 == 99)
         {
            double change_in_wji=0.0, change_in_wkj=0.0;
            vt = 0.0;
            for(int k=0;k<10;k++)
            {
               for(int j=0;j<hidden_nodes;j++)
               {
                  //cout<<vt<<"\t";
                  vt = (b2)*vt + (1-b2)*(del_Wkj[k][j]/100.0)*(del_Wkj[k][j]/100.0);
                  //eta1 = (eta1/(sqrt(vt) + epsilon));
                  change_in_wkj = b1*change_in_wkj + (1-b1)*(del_Wkj[k][j]/100.0);
                  wkj[k][j] = wkj[k][j] - (eta1/(sqrt(vt) + epsilon))*change_in_wkj;
                  del_Wkj[k][j] = 0;
               }
               //cout<<"\n";
            }
            //cout<<"\n eta1 = "<<eta1;
            //cout<<"\n";
            // updating Wji - stochastic gradient descent
            vt = 0.0;
            for(int j=0; j<hidden_nodes; j++)
            {
               for(int i=0; i<64; i++)
               {
                  vt = b2*vt + (1-b2)*(del_Wji[j][i]/100.0)*(del_Wji[j][i]/100.0);
                  //eta2 /= (sqrt(vt) + epsilon);
                  change_in_wji = b1*change_in_wji + (1-b1)*(del_Wji[j][i]/100.0);
                  wji[j][i] = wji[j][i] - (eta2/(sqrt(vt) + epsilon))*change_in_wji;
                  del_Wji[j][i]=0;
               }
            }

         }

         /*for(int k=0;k<10;k++)
         {
         for(int j=0;j<hidden_nodes;j++)
         {
         //cout<<del_Wkj[k][j]<<"\n";
         wkj[k][j] = wkj[k][j] - 0.05*del_Wkj[k][j];
      }
   }
   for(int j=0; j<hidden_nodes; j++)
   {
   for(int i=0; i<64; i++)
   {
   wji[j][i] = wji[j][i] - 0.05*del_Wji[j][i];
}
}*/
}
int error = 0;
for(int jj = 0;jj<validation.size();jj++)
{
   int max_ind;
   double ak[10], max = -1000000.0;
   int p = vv[jj];
   for(int k=0;k<10;k++)
   {
      double pp = 0.0;
      for(int j=0;j<hidden_nodes;j++)
      {
         double aj = 0.0;
         for(int i=0;i<64;i++)
         {
            aj = aj + wji[j][i]*validation[jj][i];
         }
         pp =pp + wkj[k][j]*sigmoid(aj);
      }
      ak[k] = pp;
      //cout<<pp<<"  "<<k<<endl;
      if(ak[k] > max){
         max_ind = k;
         max = ak[k];
      }
   }
   //max_ind = find_max(ak);
   if(max_ind == p) error++;;
}
cout<<epoch<<" "<<error*100/validation.size()<<"\n";
if(error>=800)
break;

//usleep(1000);
}
//print(10, hidden_nodes, wkj);
//print(wji);


read.close();
read.open("test.txt");
while(read>>s)
{
   int sz = s.size();
   vector<double> temp;
   string ss;
   for(int i=0;i<s.size()-1;i++)
   {
      if(s[i]==',')
      {
         double f = atof(ss.c_str());
         ss.clear();
         temp.pb(f);
      }
      else
      {
         ss = ss + s[i];
      }
   }
   int max_ind;
   double ak[10], max = -1000000.0;
   int p = s[sz-1]-'0';
   for(int k=0;k<10;k++)
   {
      double pp = 0.0;
      for(int j=0;j<hidden_nodes;j++)
      {
         double aj = 0.0;
         for(int i=0;i<64;i++)
         {
            aj = aj + wji[j][i]*sigmoid(temp[i]);
         }
         pp =pp + wkj[k][j]*sigmoid(aj);
      }
      ak[k] = pp;
      //cout<<pp<<"  "<<k<<endl;
      if(ak[k] > max){
         max_ind = k;
         max = ak[k];
      }
   }
   //max_ind = find_max(ak);
   if(max_ind == p) true_positive+=1;
   confusion[p][max_ind]++;
   //cout<<"****"<<p<<"***********"<<max_ind<<"****\n";
   temp.clear();
}
cout<<true_positive<<endl;
cout<<"CONFUSION MATRIX\n";
// cout<<"______________________________________________________________________________\n";
cout<<" \t";
for(int i=0;i<10;i++) cout<<i<<"\t";
cout<<"\n\n";
for(int i=0;i<10;i++)
{
   //cout<<"______________________________________________________________________________\n";
   cout<<i<<"\t";
   for(int j=0;j<10;j++)
   cout<<confusion[i][j]<<"\t";
   cout<<"\n";
}

return 0;
}
