 // Estimation of Expected Returns and Dividend Growth Rate// as in the KVB paper (2010).
 //An important point is to be noted here is that unlike the original KVB paper,the variance matrix in the kalman Filter
 //	is modeled by the Choleski decomposition
//The objective of this code is to derive the autoregressive parameters of the expected returns regression//
// Inputs : Y matrix (s_mY) - price dividend ratio, dividend growth rate.
// Output : Optimal parameters
// Dooruj Rambaccussing	
// 27 November 2009

#include <oxstd.h>
#import <database>
#import <maximize>
#include <maximize.h>
#import <packages/maxsa/maxsa>
#include <packages/maxsa/maxsa.h>

static decl mYt, mYt1, iZeta, iRho, iIota, iT   ;

mFunc(const vP)
{	 
   	decl mF = zeros(4,4);	  														//defining the F matrix
	mF[0][0]=vP[2];		  															// it implies that we would to send the parameter Vp[0] to this address after optimization
	mF[0][2]=1;
	return 	mF;
}


mRFunc(const vP)
{
	decl mR = zeros(4,3);	  														// defining the R matrix
	mR[1][0]= 1;
	mR[2][1]= 1;
	mR[3][2]= 1;
	return mR;
}																				// this is the value of K in Birbergen Pg 6)

vM0Func(const vP,const iIota,const iRho)
{
	decl iA	 = (iIota/(1-iRho))+((vP[0]-vP[1])/(1-iRho));	   						// note the vP[0] and vP[1] are parameters to be eestimated
	decl vM0 = zeros(2,1); 	 														//defining the vector M0
	vM0[0][0] = vP[0];
	vM0[1][0] =	(1-vP[3])*iA;
	return vM0;
}

mM1Func(const vP)
{
	decl mM1 = zeros(2,2);															//defining the vector M1
	mM1[1][1] = vP[3];
	return mM1;
}

mM2Func(const vP,const iRho)	
{
	decl iB1 = 1/(1-(iRho*vP[3]));
	decl iB2 = 1/(1-(iRho*vP[2]));
	decl mM2 = zeros(2,4);	 														//defining the vector M2
	mM2 [0][0] = 1;
	mM2 [0][1] = 1;
	mM2 [1][0] = iB2*(vP[2]-vP[3]);
	mM2 [1][2] = iB2;  
	mM2 [1][3] = -iB1;
	return mM2;
}

SigmaFunc(const vP)
{
//	decl Sigma;
	decl P =zeros(3,3);
//	decl sigmamug = vP[5]*vP[4]*vP[7];
//	decl sigmamuD = vP[8]*vP[5]*vP[6];
//	decl sigmagD = vP[4]*vP[6]*0;
//	Sigma = zeros(3,3);
	P[0][0] = vP[4];
	P[1][0]=  0;
	P[1][1] = vP[5];
	P[2][0]=  vP[6];
	P[2][1] = vP[7];
	P[2][2] = vP[8];
	return P*P'; 
}				  

 
KalmanFilOx(const mYt,const vP)	
{
	decl  mKF;
	decl mF, mR, vM0, mM1, mM2, Sigma;
	iZeta = new array[iT];
	iRho = new array[iT];
	iIota = new array[iT];
	decl amP = new array [iT];
	decl avX = new array[iT];
	decl avX1 = new array[iT];
	decl avvX = new array[iT];
	decl amP1 = new array [iT];
	decl ameta = new array [iT];
	decl ammP = new array [iT];
	decl amS = new array[iT];
	decl amK = new array [iT];
	avX[0] = zeros(4,1);
	amP[0]	= unit(4,4);
	ameta[0] = zeros(2,1);
	mKF = new array [iT];
	mF = mFunc(vP);
	mR = mRFunc(vP);
	mM1 = mM1Func(vP);
	for ( decl i = 1; i< iT; ++i)
   	{
		iZeta[i]= meanc(mYt[0:i][1]) ;
		iRho[i] =0.9613;
		iIota[i] = log(1+exp(iZeta[i]))- (iRho[i]*iZeta[i]);
		vM0 = vM0Func(vP,iIota[i],iRho[i]);
		mM2 = mM2Func(vP,iRho[i]);
		Sigma = SigmaFunc(vP);
		avX1[i] = avX[i-1];
		avvX[i] = mF*avX1[i];
		amP1[i] = amP[i-1];
		ammP[i] = mF*amP1[i]*mF' + mR*Sigma*mR';
		ameta[i] = (mYt[i][])' - vM0 - (mM1*(mYt1[i][])') - (mM2*avvX[i]);
		amS[i] = mM2*ammP[i]*mM2';
		amK[i] = ammP[i]*mM2'*(amS[i]^(-1));
		avX[i] = avvX[i] + amK[i]*ameta[i];
		amP[i] = (unit(4) -amK[i]*mM2)*ammP[i];
		mKF[i] = ameta[i]~amS[i];
	}
	  // print("amS =", amS);
	return mKF; 
}


Loglikelihood(const vP, const adFunc, const avScore,const amHess)
{
	decl i ,mKF;
	mKF = KalmanFilOx(mYt, vP);
	decl A = new array[iT];			
	decl B = new array[iT];
	decl H = new array[iT];
	decl K = new array[iT];
	decl L = new array[iT];
	decl X = zeros(iT,1);
	for (i = 1; i < iT; i++)			
	{
		A[i] = mKF[i][][0];		
		B[i] = mKF[i][][1:];
		H[i] = log(determinant(B[i][][]));
		H[0] = 0;
		K[i] =((A[i]'/B[i])*A[i]);
		K[0] = 0;
		L[i] = -H[i]-K[i];
		X[i][] = L[i]; 
	}
	decl M= sumc(X);
//	print("M=", M, " vP =" ,vP);
	adFunc[0] = M;	  
	return 1;
}



main()
{
   	decl vP, ir, dFunc, i,dT,dRt, iNS, iNT, vC, vM,vLo, vHi;
	decl file = "C:\Users\drambaccussing\Desktop/data1.xlsx";
	decl Get_Data = new Database();
	Get_Data.Load(file);
	decl data = Get_Data.GetAll();
	decl names = Get_Data.GetAllNames();
	mYt = data [][:1];
	mYt1 = lag0(mYt,1);
	iT = rows(mYt);





	
	vP = <0.0681295,0.108285,0.556104,0.934104,0.0288054,0.0649762,0.003,0.009, 0.0009>';
	ir= MaxBFGS(Loglikelihood, &vP, &dFunc, 1,1);
	decl hess, se;
	decl archery = Num2Derivative(Loglikelihood, vP  , &hess);
	println(archery);
	se = sqrt(diagonal(invertsym(-hess)) / rows(mYt)) ;
	println("se =", se);
  	decl sigmamu11 = ((vP[6]^2)+ (vP[7]^2)+	(vP[8]^2))^0.5 ;
	decl rhogmu = (vP[5]*vP[7])/(vP[5]*sigmamu11) ;
	decl rhomuD11 =	(vP[6]*vP[4])/(vP[4]*sigmamu11);
	println("gamma_0  =",vP[0]);
	println("delta_0  =",vP[1]);
	println("gamma_1  =",vP[2]);
	println("delta_1  =",vP[3]);
	println("sigma_d  =",vP[4]);
	println("sigma_g  =",vP[5]);
	println("sigma_mu  =",sigmamu11);
	println("rho_g_mu  =",rhogmu);
	println("rho_mu_D  =",rhomuD11);
	//
	//
	//
	println(" "," ");
	println(" "," ");
	println(" "," ");
	println("se_gamma_0  =",se[0]);
	println("se_delta_0  =",se[1]);
	println("se_gamma_1  =",se[2]);
	println("se_delta_1  =",se[3]);
	println("se_sigma_d  =",se[4]);
	println("se_sigma_g  =",se[5]);
	decl sesigmamu11 = ((se[6]^2)+ (se[7]^2)+	(se[8]^2))^0.5 ;
	//
	decl sesrhogmu = (se[5]*se[7])/(se[5]*sigmamu11) ;
	decl sesrhomuD11 =	(se[6]*se[4])/(se[4]*sigmamu11);
	println("se_sigma_mu  =",sesigmamu11);
	println("se_rho_g_mu ="	, sesrhogmu);
	println("se_rho_D_mu =", sesrhomuD11);
	
}


