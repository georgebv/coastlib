#include <math.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define	Subprograms
#define	Int	extern int
#define	Double	extern double
#include	"Headers.h"

extern double SU;

void
	Title_block(FILE*), Input_Data_block(FILE*);

/**********************************************************************/
int Read_data(void)
/**********************************************************************/
{
Readtext(Title);
iff(Title, FINISH) return(0);
Read(MaxH,lf);

if(MaxH >= 0.) // If depth is finite
	{
	strcpy(Depth,"Finite");
	fscanf(Input1,"%s", Case); Skip;
	iff(Case,Wavelength)
		{
		Read(L,lf);
		Height = MaxH/L;
		}
	iff(Case,Period)
		{
		Read(T,lf);
		Height = MaxH/(T*T);
		}
	}

if(MaxH < 0.) // If depth is infinite
	{
	strcpy(Depth,"Deep");
	fscanf(Input1,"%s", Case); Skip;
	Skip;
	Height = -MaxH;
	}

Read(Current_criterion,d);
Read(Current,lf);
if(Current_criterion == 1) strcpy(Currentname, Current1);
if(Current_criterion == 2) strcpy(Currentname, Current2);

Read(n,d);
Read(nstep,d);

Input_Data_block(monitor);

if(strcmp(Theory,"Stokes")==0)
	{
	iff(Case,Wavelength)
		if(L > 10.)
			{
			printf("\nThe dimensionless wavelength is greater than 10.");
			printf("\nStokes theory should not be applied. Exiting.");
			getch();
			exit(1);
			}
	iff(Case,Period)
		if(T > 10.)
			{
			printf("\nThe dimensionless period is greater than 10.");
			printf("\nStokes theory should not be applied. Exiting.");
			getch();
			exit(1);
			}
	}

// Convergence criteria

Input2=fopen(Convergence_file,"r");
fgets(dummy,400,Input2);
fscanf(Input2,"%d", &number);fgets(dummy,400,Input2);
fscanf(Input2,"%le", &crit);fgets(dummy,400,Input2);
fclose(Input2);

// Number of data points to present results for

Input2 = fopen(Points_file,"r");
fgets(dummy,400,Input2);
// Number of points on surface profile (clustered quadratically near crest)
fscanf(Input2,"%d", &Surface_points);fgets(dummy,400,Input2);
// Number of vertical profiles
fscanf(Input2,"%d", &Nprofiles);fgets(dummy,400,Input2);
// Number of points in each profile
fscanf(Input2,"%d", &Points);fgets(dummy,400,Input2);

fclose(Input2);

return(1);
}

//	PRINT OUT TITLE BLOCKS

void Input_Data_block(FILE* file)
{
fprintf(file,"# %s", Title);
fprintf(file,"\n\n# Printing input data here to check");
fprintf(file,"\n\n# Height/Depth:%6.3f", MaxH);
iff(Case,Wavelength)
	{
	fprintf(file,"\n# Length/Depth:%7.2f", L);
	}
iff(Case,Period)
	{
	fprintf(file,"\n# Dimensionless Period T*sqrt(g/d):%7.2f", T);
	}
fprintf(file,"\n# Current criterion: %s,  Dimensionless value:%6.3lf", Currentname, Current);

if(strcmp(Theory,"Stokes")==0)
	{
	if(n<=5) sprintf(Method, "\n# Solution by %d-order Stokes theory", n);
	else
		{
		n = 5;
		sprintf(Method, "\n# Solution by %d-order Stokes theory", n);
		printf("\n\n# (A value of N > 5 has been specified for the Stokes theory.");
		printf("\n# I do not have a theory for that. The program has set N = 5)");
		}
	}
if(strcmp(Theory,"Fourier")==0)
	sprintf(Method, "\n# Solution by %d-term Fourier series", n);

fprintf(file,"\n%s\n", Method);
}

void Title_block(FILE* file)
{
// Highest wave - eqn (32) of Fenton (1990)
L = 2*pi/z[1];
Highest = (0.0077829*L*L*L+0.0095721*L*L+0.141063*L)
	/(0.0093407*L*L*L+0.0317567*L*L+0.078834*L+1);
fprintf(file,"# %s", Title);
fprintf(file,"\n%s\n", Method);
fprintf(file,"\n# Height/Depth:%6.3f, %3.0lf\%% of the maximum of H/d =%6.3f for this length:",
	z[2]/z[1],z[2]/z[1]/Highest*100., Highest);
fprintf(file,"\n# Length/Depth:%7.2f", 2*pi/z[1]);
fprintf(file,"\n# Dimensionless Period T*sqrt(g/d):%7.2f", z[3]/sqrt(z[1]));
fprintf(file,"\n# Current criterion: %s,  Dimensionless value:%6.3lf\n", Currentname, Current);
}

void Title_block_deep(FILE* file)
{
// Highest wave
Highest = 0.141063;
fprintf(file,"# %s", Title);
fprintf(file,"\n%s\n", Method);
fprintf(file,"\n# Height/Length:%6.3f, %3.0lf\%% of the maximum of H/L =%6.3f",
	z[2]/2/pi,(z[2]/2/pi)/Highest*100., Highest);
fprintf(file,"\n# Dimensionless Period T*sqrt(g/L):%7.2f", z[3]/sqrt(2*pi));
fprintf(file,"\n# Current criterion: %s,  Dimensionless value:%6.3lf\n", Currentname, Current);
}

void Results(const char *Description, double x, double y)
{
static int Virgin=0;
Virgin++;
fprintf(Solution, "\n%s" LO, Description, x);
Is_finite
	{
	fprintf(Solution,"" LO, y);
	fprintf(Solution2,"%2d\t%15.7e\t%15.7e\t%s\n", Virgin, x, y, Description);
	}
}

void Output(void)
{
int 		i, I;
double 	X, eta, y;
double	Surface(double), Point(double, double);

fprintf(monitor,"\n\n# Solution summary:\n\n");
Is_finite Title_block(monitor);
Is_deep Title_block_deep(monitor);

// Print out summary file of solution

Is_finite Title_block(Solution);
Is_deep Title_block_deep(Solution);

Is_finite
	{
	kd = z[1];
	L=2*pi/z[1];
	H=z[2]/z[1];
	T=z[3]/sqrt(z[1]);
	c=z[4]/sqrt(z[1]);
	ce=z[5]/sqrt(z[1]);
	cs=z[6]/sqrt(z[1]);
	ubar=z[7]/sqrt(z[1]);
	Q=ubar-z[8]/pow(kd,1.5);
	R=1+z[9]/z[1];

	pulse=z[8]+z[1]*z[5];
	ke=0.5*(z[4]*pulse-z[5]*Q*pow(kd,1.5));

	// Calculate potential energy, not by computing the mean of 1/2 (eta-d)^2
	// but by exploiting orthogonality of the cosine functions to give the sum of 1/4 Y[i]^2
	pe = 0;
	for(i=1;i<=n;++i)
		pe += 0.25*pow(Y[i],2);

	ub2=2.*z[9]-z[4]*z[4];
	sxx=4.*ke-3.*pe+ub2*z[1]+2.*z[5]*(z[7]*z[1]-z[8]);
	f=z[4]*(3.*ke-2.*pe)+0.5*ub2*(pulse+z[4]*z[1])+z[4]*z[5]*(z[7]*z[1]-z[8]);
	q=z[7]*z[1]-z[8];
	r=z[9]+z[1];
	s=sxx-2.*z[4]*pulse+(z[4]*z[4]+0.5*z[1])*z[1];
	}

Is_finite fprintf(Solution, "\n# Stokes-Ursell number %7.3f", 0.5*z[2]/pow(z[1],3));
fprintf(Solution, "\n\n# Integral quantities - notation from Fenton (1988)");
fprintf(Solution2,"# %s", Title);
fprintf(Solution2,"\n# Solution non-dimensionalised by (1) g & wavenumber, and (2) g & mean depth\n");
Is_finite
	{
	fprintf(Solution, "\n# (1) Quantity, (2) symbol, solution non-dimensionalised by (3) g & wavenumber, and (4) g & mean depth\n");
	//fprintf(Solution, "\n# Water depth                        (d)"); Results(z[1], 1.);
	Results("# Water depth                        (d)", z[1], 1.);
	}
Is_deep
	{
	fprintf(Solution, "\n# (1) Quantity, (2) symbol, solution non-dimensionalised by (3) g & wavenumber\n");
	}
Results("# Wave length                   (lambda)", 2*pi, L);
Results("# Wave height                        (H)", z[2], H);
Results("# Wave period                      (tau)", z[3], T);
Results("# Wave speed                         (c)", z[4], c);
Results("# Eulerian current                 (u1_)", z[5], ce);
Results("# Stokes current                   (u2_)", z[6], cs);
Results("# Mean fluid speed in frame of wave (U_)", z[7], ubar);
Results("# Volume flux due to waves           (q)", z[8], z[8]/pow(kd,1.5));
Results("# Bernoulli constant                 (r)", z[9], z[9]/kd);

Is_finite
	{
	Results("# Volume flux                        (Q)", Q*pow(kd,1.5), Q);
	Results("# Bernoulli constant                 (R)", R*kd, R);
	Results("# Momentum flux                      (S)", s, s/kd/kd );
	Results("# Impulse                            (I)", pulse, pulse/pow(kd,1.5));
	Results("# Kinetic energy                     (T)", ke, ke/kd/kd);
	Results("# Potential energy                   (V)", pe, pe/kd/kd);
	Results("# Mean square of bed velocity     (ub2_)", ub2, ub2/kd);
	Results("# Radiation stress                 (Sxx)", sxx, sxx/kd/kd);
	Results("# Wave power                         (F)", f, f/pow(kd,2.5));
	}

fprintf(Solution, "\n\n# Dimensionless coefficients in Fourier series" );
fprintf(Solution, "\n# Potential/Streamfn\tSurface elevations" );
fprintf(Solution, "\n# j, B[j], & E[j], j=1..N\n" );
fprintf(Solution2, "%2d\t# N, number of dimensionless Fourier coefficients - j, B[j], & E[j] below\n", n);

for ( i=1 ; i <= n ; i++ )
	{
	fprintf(Solution, "\n%2d\t%15.7e\t%15.7e", i, B[i], Y[i]);
	fprintf(Solution2, "%2d\t%15.7e\t%15.7e\n", i, B[i], Y[i]);
	}
fprintf(Solution, "\n\n" );

// Surface - print out coordinates of points on surface for plotting plus check of pressure on surface

fprintf(Elevation,  "# %s\n", Title);
fprintf(Elevation,  "%s\n", Method);
fprintf(Elevation,  "\n# Surface of wave - trough-crest-trough,");
fprintf(Elevation,  " note quadratic point spacing clustered around crest");
Is_finite
	{
	fprintf(Elevation,  "\n# Non-dimensionalised with respect to depth");
	fprintf(Elevation,  "\n# X/d, eta/d, & check of surface pressure\n");
	fprintf(Elevation,  "\n0.\t0.\t0. # Dummy point to scale plot\n");
	}
Is_deep
	{
	fprintf(Elevation,  "\n# Non-dimensionalised with respect to wavenumber");
	fprintf(Elevation,  "\n# kX, k eta, & check of surface pressure\n");
	}

for ( i=-Surface_points/2 ; i <= Surface_points/2; i++)
	{
	X = 4 * pi * (i * fabs(i)/Surface_points/Surface_points);	//NB Quadratic point spacing, clustered near crest
	eta = Surface(X);
	Point(X,eta);
   Is_finite fprintf(Elevation,  "\n%8.4lf\t%7.4f\t%7.0e", X/kd, 1+eta/kd, Pressure);
	Is_deep   fprintf(Elevation,  "\n%8.4lf\t%7.4f\t%7.0e", X, eta, Pressure);
	}
fprintf(Elevation,  "\n\n");

// Surface - print out Velocity and acceleration profiles plus check of Bernoulli

fprintf(Flowfield,  "# %s\n", Title);
fprintf(Flowfield,  "%s\n", Method);
fprintf(Flowfield,  "\n# Velocity and acceleration profiles and Bernoulli checks\n");
fprintf(Flowfield,  "\n# All quantities are dimensionless with respect to g and/or ");
Is_deep fprintf(Flowfield,  "k\n");
Is_finite fprintf(Flowfield,  "d\n");
fprintf(Flowfield,  "\n#*******************************************************************************");
Is_finite
	{
	fprintf(Flowfield,  "\n# y        u       v    dphi/dt   du/dt   dv/dt  du/dx   du/dy Bernoulli check  ");
	fprintf(Flowfield,  "\n# -     -------------   -------  ------   -----  ------------- ---------------  ");
	fprintf(Flowfield,  "\n# d        sqrt(gd)       gd        g       g       sqrt(g/d)        gd         ");
	}
Is_deep
	{
	fprintf(Flowfield,  "\n# ky       u       v    dphi/dt   du/dt   dv/dt  du/dx   du/dy Bernoulli check  ");
	fprintf(Flowfield,  "\n#       -------------   -------  ------   -----  ------------- ---------------  ");
	fprintf(Flowfield,  "\n#         sqrt(g/k)       g/k       g       g       sqrt(gk)        g/k         ");
	}
fprintf(Flowfield,  "\n#*******************************************************************************");

fprintf(Flowfield,  "\n# Note that increasing X/d and 'Phase' here describes half of a wave for");
fprintf(Flowfield,  "\n# X/d >= 0. In a physical problem, where we might have a constant x, because");
fprintf(Flowfield,  "\n# the phase X = x - c * t, then as time t increases, X becomes increasingly");
fprintf(Flowfield,  "\n# negative and not positive as passing down the page here implies.");

for(I = 0; I <= Nprofiles ; ++I)
	{
	X = pi * I/(Nprofiles);
	eta = Surface(X);
	fprintf(Flowfield,  "\n\n# X/d = %8.4f, Phase = %6.1f°\n", X/kd, X*180./pi);

	for(i=0 ; i <= Points; ++i)
		{
		Is_finite
			{
			y = (i)*(1.+eta/kd)/(Points);
			Point(X, kd*(y-1.));
			}
		Is_deep
			{
			y = -pi + (double)i/Points*(eta+pi);
			Point(X,y);
			}
		fprintf(Flowfield,  "\n%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f\t%7.4f",
			y, u, v, dphidt, ut, vt, ux, uy, Bernoulli_check);
		}
	}
fprintf(Flowfield,  "\n\n");

/*
Procedure for recording every run - not activated in distribution versions
If the lines below are not commented out the program will add a line to a
file Catalogue.res, which could have this as a header:

# A continuing record of all runs with Fourier, Cnoidal, or Stokes.
# Any run of those programs adds a line to it.
# This can be edited at any time.
# Columns are: name of theory, N, H/d, L/d, Stokes-Ursell Number,
# wave height as a percentage of the highest possible for that L/d,
# mean horizontal velocity on a vertical line under the crest.

# Theory n   H/d       L/d     S-U Highest% u_crest_mean
*/

// To activate, de-comment these lines
/***************************************************************************
FILE *Output1;
double Velo[Points+1], sum1, sum2, ucm;
Output1 = fopen(Diagname,"a");

X = 0.;
eta = Surface(X);

for(i=0 ; i <= Points; ++i)
	{
	y = (i)*eta/(Points);
	Point(X, y);
	Velo[i] = u;
	}

for(i=1, sum1=0; i <= Points-1; i+=2) sum1 += Velo[i];
for(i=2, sum2=0; i <= Points-2; i+=2) sum2 += Velo[i];
ucm = (Velo[0]+4*sum1+2*sum2+Velo[Points])/3./Points;

I = strlen(Theory)+1;
for(i=I; i <=8 ; ++i) strcat(Theory," ");
fprintf(Output1,"\n%s%2d\t%7.4f\t%8.3f\t%7.3f\t%3.0f\t%7.4f",
			Theory, n, H, L, 0.5*z[2]/pow(z[1],3), z[2]/z[1]/Highest*100., ucm);
*************************************************************************/
// All this is for the diagrams

//Point(0.,z[10]);
//printf("\nPsi %g",psi);

double psi0, psi1,Um,dd;

Point(0.,Surface(0.));
psi1 = psi;
Point(0.,-kd);
psi0 = psi;

//if(theory==0)
//	SU = 0.5*z[2]/pow(z[1],3);

dd = 1.+Surface(0.)/kd;
Um = (psi1-psi0)/dd;
printf("\nUm %g", Um);
}

// Surface elevation

double Surface(double x)
{
int j;
static double kEta;

kEta = 0.;
for ( j = 1 ; j < n ; j++ )
	kEta += Y[j] * cos(j*x);
kEta += 0.5 * Y[n] * cos(n*x);
return (kEta);
}

// Velocities, accelerations, and pressure at a point

void Point(double X, double Y)
{
int j;
double C, S, Sin, Cos, y;
double coshdelta,sinhdelta;

u = v = ux = vx = phi = psi = 0.;
y = 1.+Y/kd;

for ( j = 1 ; j <= n ; j++ )
	{
   Cos  = cos(j*X);
	Sin  = sin(j*X);
	Is_finite
		{
		coshdelta = cosh(j*Y);
		sinhdelta = sinh(j*Y);
		C = coshdelta+sinhdelta*Tanh[j];
		S = sinhdelta+coshdelta*Tanh[j];
		}
	Is_deep
		C = S = exp(j*Y);
	phi += B[j] * C * Sin;
	psi += B[j] * S * Cos;
	u += j * B[j] * C * Cos;
	v += j * B[j] * S * Sin;
	ux += - j * j * B[j] * C * Sin;
	vx += j * j * B[j] * S * Cos;
	}

Is_finite
	{
	// All PHI, PSI, u, v, ux and vx are dimensionless w.r.t. g & k.
	// Now convert to dimensionless w.r.t. d.
	phi /= pow(kd,1.5);
	psi /= pow(kd,1.5);
	u /= pow(kd,0.5);
	v /= pow(kd,0.5);
	ux *= pow(kd,0.5);
	vx *= pow(kd,0.5);
	u = ce + u;
	phi = ce * X + phi;
	psi = ce * y + psi;
	dphidt = -c * u;

	ut = -c * ux;
	vt = -c * vx;
	uy = vx;
	vy = -ux;
	dudt = ut + u*ux + v*uy;
	dvdt = vt + u*vx + v*vy;
	Pressure = R - y - 0.5 * ((u-c)*(u-c)+v*v);
	Bernoulli_check = dphidt + Pressure + y + 0.5*(u*u+v*v)-(R-0.5*c*c);
	//printf("\n%f %f %f %f %f", R, y, 0.5*((u-c)*(u-c)+v*v),Pressure,Bernoulli_check);
	}

Is_deep
	{
	u = z[5] + u;
	phi = z[5] * X + phi;
	dphidt = -z[4] * u;

	ut = -z[4] * ux;
	vt = -z[4] * vx;
	uy = vx;
	vy = -ux;
	dudt = ut + u*ux + v*uy;
	dvdt = vt + u*vx + v*vy;
	Pressure = z[9] - Y - 0.5 * ((u-z[4])*(u-z[4])+v*v);
	Bernoulli_check = dphidt + Pressure + Y + 0.5*(u*u+v*v)-(z[9]-0.5*z[4]*z[4]);
	}
return;
}

