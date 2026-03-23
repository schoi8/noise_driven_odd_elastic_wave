#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*
	analyze the statistics of the noise in the starfish embryo model
	one realization
	get the current correlation function as well as the position and velocity. We get the C for just one realization to mimic the expt.
*/

#define numx 30 // number of columns. has to be even for pbc.
#define numy 30 // number of rows. has to be even for pbc.
#define N numx*numy // number of particles.
//#define f0 0.06 // amplitude of transverse force
#define f0 1.0

double getDist(double xi, double xj, double L);
double getPBCval(double x, double L);
double drdt_xl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj);
double drdt_yl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj);
double nearfield(double dx, double dy);
double drdt_xt(double dx, double dy, double wi, double wj);
double drdt_yt(double dx, double dy, double wi, double wj);
int* getNeighIdx(int i);

int main() {
	clock_t tStart = clock();
	srand((unsigned)time(NULL));

	// preparing to generate normal random variables using gsl
	const gsl_rng_type* T_gauss;
	gsl_rng* gr;
	gsl_rng_env_setup();
	T_gauss = gsl_rng_default;
	gr = gsl_rng_alloc(T_gauss);
	gsl_rng_set(gr, time(NULL)); // seed based on the current time

	double sig = 1.0; // std for Gaussian randon variable

	// simulation time
	int n_tot = 10000; // total number of time steps
	int n_interval = 10; // time step interval for data recording
	int n_rec = (int)(n_tot / n_interval);
	double tcur = 0.0;
	double dt = 0.01;

	char basedir[] = "/directory_to_store_data/";
	char txt[] = ".txt";
	char dat[] = ".dat";

	char file_para[] = "simulation_para";
	char dir_para[100];
	sprintf(dir_para, "%s%s%s", basedir, file_para, txt);

	char file_xarr[] = "xarr";
	char dir_xarr[100];
	sprintf(dir_xarr, "%s%s%s", basedir, file_xarr, dat);
	char file_yarr[] = "yarr";
	char dir_yarr[100];
	sprintf(dir_yarr, "%s%s%s", basedir, file_yarr, dat);
	char file_vxarr[] = "vxarr";
	char dir_vxarr[100];
	sprintf(dir_vxarr, "%s%s%s", basedir, file_vxarr, dat);
	char file_vyarr[] = "vyarr";
	char dir_vyarr[100];
	sprintf(dir_vyarr, "%s%s%s", basedir, file_vyarr, dat);
	char file_tharr[] = "tharr";
	char dir_tharr[100];
	sprintf(dir_tharr, "%s%s%s", basedir, file_tharr, dat);

	char file_Cll[] = "Cll_starfish";
	char dir_Cll[100];
	sprintf(dir_Cll, "%s%s%s", basedir, file_Cll, dat);
	char file_Cltr[] = "Cltr_starfish";
	char dir_Cltr[100];
	sprintf(dir_Cltr, "%s%s%s", basedir, file_Cltr, dat);
	char file_Clti[] = "Clti_starfish";
	char dir_Clti[100];
	sprintf(dir_Clti, "%s%s%s", basedir, file_Clti, dat);
	char file_Ctt[] = "Ctt_starfish";
	char dir_Ctt[100];
	sprintf(dir_Ctt, "%s%s%s", basedir, file_Ctt, dat);

	FILE* fp_xarr = fopen(dir_xarr, "w");
	fclose(fp_xarr);
	FILE* fp_yarr = fopen(dir_yarr, "w");
	fclose(fp_yarr);
	FILE* fp_vxarr = fopen(dir_vxarr, "w");
	fclose(fp_vxarr);
	FILE* fp_vyarr = fopen(dir_vyarr, "w");
	fclose(fp_vyarr);
	FILE* fp_tharr = fopen(dir_tharr, "w");
	fclose(fp_tharr);

	char file_xcur[] = "xcur";
	char dir_xcur[100];
	sprintf(dir_xcur, "%s%s%s", basedir, file_xcur, txt);
	char file_ycur[] = "ycur";
	char dir_ycur[100];
	sprintf(dir_ycur, "%s%s%s", basedir, file_ycur, txt);
	char file_vxcur[] = "vxcur";
	char dir_vxcur[100];
	sprintf(dir_vxcur, "%s%s%s", basedir, file_vxcur, txt);
	char file_vycur[] = "vycur";
	char dir_vycur[100];
	sprintf(dir_vycur, "%s%s%s", basedir, file_vycur, txt);
	char file_thcur[] = "thcur";
	char dir_thcur[100];
	sprintf(dir_thcur, "%s%s%s", basedir, file_thcur, txt);

	FILE* fp_xcur = fopen(dir_xcur, "w");
	fclose(fp_xcur);
	FILE* fp_ycur = fopen(dir_ycur, "w");
	fclose(fp_ycur);
	FILE* fp_vxcur = fopen(dir_vxcur, "w");
	fclose(fp_vxcur);
	FILE* fp_vycur = fopen(dir_vycur, "w");
	fclose(fp_vycur);
	FILE* fp_thcur = fopen(dir_thcur, "w");
	fclose(fp_thcur);

	char file_vxnoise[] = "vxnoise";
	char dir_vxnoise[100];
	sprintf(dir_vxnoise, "%s%s%s", basedir, file_vxnoise, dat);
	char file_vynoise[] = "vynoise";
	char dir_vynoise[100];
	sprintf(dir_vynoise, "%s%s%s", basedir, file_vynoise, dat);

	FILE* fp_vxnoise = fopen(dir_vxnoise, "w");
	fclose(fp_vxnoise);
	FILE* fp_vynoise = fopen(dir_vynoise, "w");
	fclose(fp_vynoise);

	double pi = acos(-1); // define pi

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.2; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing
	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.
	double rm = 3.8 * R; // neighbor detecting radius 

	double Fst0 = 53.7;
	double frep0 = 785.1;
	//double Fst0 = 0.0;
	//double frep0 = 0.0;
	double w0 = 1.0;
	//double w0 = 0.72 * 2 * pi; // single isolated embryo self-spinning freq
	//double w0 = 0.03;

	// rondom variable v0
	double vi; // "v0 value" of each particle i
	double v0 = 0.01; // average amplitude of self-propelling velocity
	//double v0 = 0.0;
	double ve = 0.1; // noise amplitude of v0.
	//double ve = 0.0;
	//double ve = sqrt(2* 10) * 0.007; // sqrt(2*expt_time_interval)*expt_sigma where expt_time_interval = 10s
	//double ve = 0.05;

	double* vxnoise;
	vxnoise = (double*)malloc(N * sizeof(double));
	double* vynoise;
	vynoise = (double*)malloc(N * sizeof(double));

	// current position
	double* xcur;
	xcur = (double*)malloc(N * sizeof(double));
	double* ycur;
	ycur = (double*)malloc(N * sizeof(double));

	// current velocity of each particle
	double* vxcur;
	vxcur = (double*)malloc(N * sizeof(double));
	double* vycur;
	vycur = (double*)malloc(N * sizeof(double));

	// angle of self-propelling motion of each particle
	double* thcur;
	thcur = (double*)malloc(N * sizeof(double));
	double thcur0; 

	// miscellaneous indices 
	int i, j, n;

	// perfect triangular lattice
	double* xlatt;
	xlatt = (double*)malloc(N * sizeof(double));
	double* ylatt;
	ylatt = (double*)malloc(N * sizeof(double));

	double u0 = 0.05; // initial displacement amplitude

	for (j = 0; j < (int)(numy / 2); j++) {
		for (i = 0; i < numx; i++) {
			xlatt[numx * 2 * j + i] = i * r0;
			xlatt[numx * (2 * j + 1) + i] = (i + 0.5) * r0;
		}
	}
	for (j = 0; j < numy; j++) {
		for (i = 0; i < numx; i++) {
			ylatt[numx * j + i] = 0.5 * sqrt(3) * j * r0;
		}
	}

	// initial condition
	for (i = 0; i < N; i++) {
		xcur[i] = xlatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		ycur[i] = ylatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		thcur[i] = 2 * pi * (double)rand() / (double)RAND_MAX; // random angle from 0 to 2pi
	}

	double dx = 0.3 * spacing;
	double dy = 0.3 * spacing;
	const int xnum = (int)round(Lx / dx);
	const int ynum = (int)round(Ly / dy);

	const int n_j = (int)((n_tot / n_interval) * 0.8);

	int wnum = (int)floor(n_j / 2) + 1;

	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky
	double dw = 2 * pi / (dt * n_interval * n_j); // resolution in w

	// kx and ky range
	double* kxrange;
	kxrange = (double*)malloc(xnum * sizeof(double));
	double* kyrange;
	kyrange = (double*)malloc(ynum * sizeof(double));

	for (i = 0; i < xnum; i++) {
		kxrange[i] = -pi / dx + dkx * i;
	}
	for (j = 0; j < ynum; j++) {
		kyrange[j] = -pi / dy + dky * j;
	}

	// w range
	double* wrange;
	wrange = (double*)malloc(wnum * sizeof(double));

	for (i = 0; i < wnum; i++) {
		wrange[i] = i * dw;
	}

	// current matrix
	double* jxmat;
	jxmat = (double*)calloc(xnum * ynum * n_j, sizeof(double));
	double* jymat;
	jymat = (double*)calloc(xnum * ynum * n_j, sizeof(double));

	int ki_shift, kj_shift;
	double jxre, jxim, jyre, jyim;
	double qxhat, qyhat, qx, qy;
	double jlre, jlim, jtre, jtim;

	double* Cll; // current correlation Jl*Jl
	Cll = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_r; // current correlation Jl*Jt
	Clt_r = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_i; // current correlation Jt*Jl
	Clt_i = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Ctt; // current correlation Jt*Jt
	Ctt = (double*)calloc(xnum * ynum * wnum, sizeof(double));

	FILE* fp_para = fopen(dir_para, "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_tot = %i, n_interval = %i, dt = %f \n", n_tot, n_interval, dt);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "R = %f, spacing = %f, r0 = %f, rm = %f \n", R, spacing, r0, rm);
	fprintf(fp_para, "Fst0 = %f, frep0 = %f, w0 = %f, f0 = %f, u0 = %f, v0 = %f \n", Fst0, frep0, w0, f0, u0, v0);
	fprintf(fp_para, "noisy v0. mean = %f, noise amplitude ve = %f \n", v0, ve);
	fclose(fp_para);

	double xij, yij, rij, xcur0, ycur0;
	int idx_neigh;
	int n_idx, ix, iy, ki, kj, ni;

	int endcode = 0;

	for (n = 0; n < n_tot; n++) {
		for (i = 0; i < N; i++) {
			vxcur[i] = 0.0;
			vycur[i] = 0.0;

			int* neighvec = getNeighIdx(i); // array of neighbor indices

			for (j = 0; j < 6; j++) {
				idx_neigh = neighvec[j];
				xij = getDist(xcur[i], xcur[idx_neigh], Lx);
				yij = getDist(ycur[i], ycur[idx_neigh], Ly);
				rij = sqrt(xij * xij + yij * yij);

				if (rij < 2 * R) {
					printf("overlap happened at time %f which is time step %i \n", tcur, n);
					printf("overlap between %d and %d \n", i, j);
					endcode = 1;
					printf("x values are %f and %f \n", xcur[i], xcur[idx_neigh]);
					printf("y values are %f and %f \n", ycur[i], ycur[idx_neigh]);
					printf("calculated xdis %f \n", xij);
					printf("calculated ydis %f \n", yij);
					printf("Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
					break;
				}
				else if (rij < rm) {
					vxcur[i] += drdt_xl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_xt(xij, yij, w0, w0);
					vycur[i] += drdt_yl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_yt(xij, yij, w0, w0);
				}
			}
			free(neighvec);
			if (endcode == 1) {
				break;
			}
		}
		if (endcode == 1) {
			break;
		}

		// update variables
		for (i = 0; i < N; i++) {
			vi = ve * gsl_ran_gaussian(gr, sig) + v0;

			vxnoise[i] = (vi - v0) * cos(thcur[i]);
			vynoise[i] = (vi - v0) * sin(thcur[i]);
			
			vxcur[i] += vi * cos(thcur[i]);
			vycur[i] += vi * sin(thcur[i]);

			xcur0 = xcur[i] + dt * vxcur[i];
			xcur[i] = getPBCval(xcur0, Lx);
			
			ycur0 = ycur[i] + dt * vycur[i];
			ycur[i] = getPBCval(ycur0, Ly);

			thcur0 = thcur[i] + dt * w0;
			thcur[i] = getPBCval(thcur0, 2 * pi);
		}
		
		// save the variables
		if (n % n_interval == n_interval - 1) {
			n_idx = (int)floor(n / n_interval);

			if (n_idx >= n_rec - n_j) {
				for (i = 0; i < N; i++) {
					ix = (int)floor(xcur[i] / dx);
					iy = (int)floor(ycur[i] / dy);

					jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i];
					jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i];
					//jxmat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vxcur[i] - vxnoise[i];
					//jymat[ix * ynum * n_j + iy * n_j + n_idx - (n_rec - n_j)] += vycur[i] - vynoise[i];
					/*
						If we calculate the current correlation function with the noise included in the velocity, we need more statistics
						to get a clear picture of the current correlation function.
					*/
				}
			}

			fp_xarr = fopen(dir_xarr, "a");
			fp_yarr = fopen(dir_yarr, "a");
			fp_vxarr = fopen(dir_vxarr, "a");
			fp_vyarr = fopen(dir_vyarr, "a");
			fp_tharr = fopen(dir_tharr, "a");
			fp_vxnoise = fopen(dir_vxnoise, "a");
			fp_vynoise = fopen(dir_vynoise, "a");

			for (i = 0; i < N; i++) {
				if (fp_xarr == NULL) {
					perror("Error in x_arr\n");
					return 1;
				}
				fprintf(fp_xarr, "%f\n", xcur[i]);

				if (fp_yarr == NULL) {
					perror("Error in y_arr\n");
					return 1;
				}
				fprintf(fp_yarr, "%f\n", ycur[i]);

				if (fp_vxarr == NULL) {
					perror("Error in vx_arr\n");
					return 1;
				}
				fprintf(fp_vxarr, "%f\n", vxcur[i]);

				if (fp_vyarr == NULL) {
					perror("Error in vy_arr\n");
					return 1;
				}
				fprintf(fp_vyarr, "%f\n", vycur[i]);

				if (fp_tharr == NULL) {
					perror("Error in th_arr\n");
					return 1;
				}
				fprintf(fp_tharr, "%f\n", thcur[i]);

				if (fp_vxnoise == NULL) {
					perror("Error in vxnoise\n");
					return 1;
				}
				fprintf(fp_vxnoise, "%f\n", vxnoise[i]);

				if (fp_vynoise == NULL) {
					perror("Error in vynoise\n");
					return 1;
				}
				fprintf(fp_vynoise, "%f\n", vynoise[i]);
			}
			fclose(fp_xarr);
			fclose(fp_yarr);
			fclose(fp_vxarr);
			fclose(fp_vyarr);
			fclose(fp_tharr);
			fclose(fp_vxnoise);
			fclose(fp_vynoise);
		}
		else if (n % n_interval == 0) {
			fp_xcur = fopen(dir_xcur, "w");
			fp_ycur = fopen(dir_ycur, "w");
			fp_vxcur = fopen(dir_vxcur, "w");
			fp_vycur = fopen(dir_vycur, "w");
			fp_thcur = fopen(dir_thcur, "w");

			for (i = 0; i < N; i++) {
				if (fp_xcur == NULL) {
					perror("Error in xcur\n");
					return 1;
				}
				fprintf(fp_xcur, "%f\n", xcur[i]);

				if (fp_ycur == NULL) {
					perror("Error in ycur\n");
					return 1;
				}
				fprintf(fp_ycur, "%f\n", ycur[i]);

				if (fp_vxcur == NULL) {
					perror("Error in vxcur\n");
					return 1;
				}
				fprintf(fp_vxcur, "%f\n", vxcur[i]);

				if (fp_vycur == NULL) {
					perror("Error in vycur\n");
					return 1;
				}
				fprintf(fp_vycur, "%f\n", vycur[i]);

				if (fp_thcur == NULL) {
					perror("Error in thcur\n");
					return 1;
				}
				fprintf(fp_thcur, "%f\n", thcur[i]);
			}
			fclose(fp_xcur);
			fclose(fp_ycur);
			fclose(fp_vxcur);
			fclose(fp_vycur);
			fclose(fp_thcur);
		}
		else {
			fp_xcur = fopen(dir_xcur, "a");
			fp_ycur = fopen(dir_ycur, "a");
			fp_vxcur = fopen(dir_vxcur, "a");
			fp_vycur = fopen(dir_vycur, "a");
			fp_thcur = fopen(dir_thcur, "a");
			
			for (i = 0; i < N; i++) {
				if (fp_xcur == NULL) {
					perror("Error in xcur\n");
					return 1;
				}
				fprintf(fp_xcur, "%f\n", xcur[i]);

				if (fp_ycur == NULL) {
					perror("Error in ycur\n");
					return 1;
				}
				fprintf(fp_ycur, "%f\n", ycur[i]);

				if (fp_vxcur == NULL) {
					perror("Error in vxcur\n");
					return 1;
				}
				fprintf(fp_vxcur, "%f\n", vxcur[i]);

				if (fp_vycur == NULL) {
					perror("Error in vycur\n");
					return 1;
				}
				fprintf(fp_vycur, "%f\n", vycur[i]);

				if (fp_thcur == NULL) {
					perror("Error in thcur\n");
					return 1;
				}
				fprintf(fp_thcur, "%f\n", thcur[i]);
			}
			fclose(fp_xcur);
			fclose(fp_ycur);
			fclose(fp_vxcur);
			fclose(fp_vycur);
			fclose(fp_thcur);
		}
		tcur += dt;
	}
	free(xlatt);
	free(ylatt);
	free(xcur);
	free(ycur);
	free(vxcur);
	free(vycur);
	free(thcur);
	gsl_rng_free(gr); // free the random number generator
	free(vxnoise);
	free(vynoise);

	if (endcode == 1) {
		free(jxmat);
		free(jymat);
		free(kxrange);
		free(kyrange);
		free(wrange);
		free(Cll);
		free(Clt_r);
		free(Clt_i);
		free(Ctt);
		exit(0);
	}

	fftw_complex* outjx;
	outjx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjx;
	pjx = fftw_plan_dft_r2c_3d(xnum, ynum, n_j, jxmat, outjx, FFTW_ESTIMATE);
	fftw_execute(pjx);
	fftw_destroy_plan(pjx);
	free(jxmat);

	fftw_complex* outjy;
	outjy = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjy;
	pjy = fftw_plan_dft_r2c_3d(xnum, ynum, n_j, jymat, outjy, FFTW_ESTIMATE);
	fftw_execute(pjy);
	fftw_destroy_plan(pjy);
	free(jymat);

	for (ki = 0; ki < xnum; ki++) {
		// shift the order in kx and ky axis because of the artifact of DFT
		if (ki < (int)(xnum / 2)) {
			ki_shift = ki + (int)(xnum / 2);
		}
		else {
			ki_shift = ki - (int)(xnum / 2);
		}

		for (kj = 0; kj < ynum; kj++) {
			qx = kxrange[ki];
			qy = kyrange[kj];

			if (qx == 0.0 && qy == 0.0) {
				qxhat = 0.0;
				qyhat = 0.0;
			}
			else {
				qxhat = qx / sqrt(qx * qx + qy * qy);
				qyhat = qy / sqrt(qx * qx + qy * qy);
			}

			// shift the order in kx and ky axis because of the artifact of DFT
			if (kj < (int)(ynum / 2)) {
				kj_shift = kj + (int)(ynum / 2);
			}
			else {
				kj_shift = kj - (int)(ynum / 2);
			}

			for (ni = 0; ni < wnum; ni++) {
				jxre = outjx[ki_shift * ynum * wnum + kj_shift * wnum + ni][0]; // real part of Jx
				jxim = outjx[ki_shift * ynum * wnum + kj_shift * wnum + ni][1]; // imaginary part of Jx
				jyre = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][0];
				jyim = outjy[ki_shift * ynum * wnum + kj_shift * wnum + ni][1];

				jlre = jxre * qxhat + jyre * qyhat; // longitudinal Jl real part
				jlim = jxim * qxhat + jyim * qyhat; // longitudinal Jl imaginary part
				jtre = jyre * qxhat - jxre * qyhat; // transverse Jt real part
				jtim = jyim * qxhat - jxim * qyhat; // transverse Jt imaginary part

				Cll[ki * ynum * wnum + kj * wnum + ni] += (jlre * jlre + jlim * jlim); // longitudinal current correalation function
				Clt_r[ki * ynum * wnum + kj * wnum + ni] += (jlre * jtre + jlim * jtim); // real part of cross current correlation function
				Clt_i[ki * ynum * wnum + kj * wnum + ni] += (jlre * jtim - jlim * jtre); // imaginary part of cross current correlation function
				Ctt[ki * ynum * wnum + kj * wnum + ni] += (jtre * jtre + jtim * jtim); // transverse current correlation function
			}
		}
	}
	fftw_free(outjx);
	fftw_free(outjy);

	free(kxrange);
	free(kyrange);
	free(wrange);

	FILE* fp_Cll;
	fp_Cll = fopen(dir_Cll, "w");
	FILE* fp_Cltr;
	fp_Cltr = fopen(dir_Cltr, "w");
	FILE* fp_Clti;
	fp_Clti = fopen(dir_Clti, "w");
	FILE* fp_Ctt;
	fp_Ctt = fopen(dir_Ctt, "w");

	for (i = 0; i < xnum * ynum * wnum; i++) {
		fwrite(&Cll[i], sizeof(double), 1, fp_Cll);
		fwrite(&Clt_r[i], sizeof(double), 1, fp_Cltr);
		fwrite(&Clt_i[i], sizeof(double), 1, fp_Clti);
		fwrite(&Ctt[i], sizeof(double), 1, fp_Ctt);
	}
	fclose(fp_Cll);
	fclose(fp_Cltr);
	fclose(fp_Clti);
	fclose(fp_Ctt);

	free(Cll);
	free(Clt_r);
	free(Clt_i);
	free(Ctt);
	
	fp_para = fopen(dir_para, "a");
	fprintf(fp_para, "Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	fclose(fp_para);
	printf("Simulation done. Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
	return 0;
}

// function to calculate the distance for pbc
double getDist(double xi, double xj, double L) {
	double dist0;
	if (xi > xj) {
		dist0 = xi - xj;
	}
	else {
		dist0 = xj - xi;
	}

	double dist;
	if (2 * dist0 <= L) {
		dist = xi - xj;
	}
	else {
		if (xi < xj) {
			dist = xi - xj + L;
		}
		else {
			dist = xi - xj - L;
		}
	}

	return dist;
}

// function to get the position in pbc
double getPBCval(double x, double L) {
	double x_pbc;

	if (x < 0.0) {
		x_pbc = x + L;
	}
	else if (x > L) {
		x_pbc = x - L;
	}
	else {
		x_pbc = x;
	}
	return x_pbc;
}

// function to calculate longitudinal force
double drdt_xl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj) {
	double pi = acos(-1.0); // define pi

	double rij = sqrt(dx * dx + dy * dy);
	double rijm = sqrt(dx * dx + dy * dy + 1);

	double Fst = 0.5 * (Fsti + Fstj);
	double frep = 0.5 * (frepi + frepj);

	double vstx = -(Fst / (8 * pi)) * dx / pow(rijm, 3);
	double frepx = (3 / pow(2, 10)) * frep * dx / pow(rij, 14);

	return (vstx + frepx);
}

double drdt_yl(double dx, double dy, double Fsti, double frepi, double Fstj, double frepj) {
	double pi = acos(-1.0); // define pi

	double rij = sqrt(dx * dx + dy * dy);
	double rijm = sqrt(dx * dx + dy * dy + 1);

	double Fst = 0.5 * (Fsti + Fstj);
	double frep = 0.5 * (frepi + frepj);

	double vsty = -(Fst / (8 * pi)) * dy / pow(rijm, 3);
	double frepy = (3 / pow(2, 10)) * frep * dy / pow(rij, 14);

	return (vsty + frepy);
}

// function to calculate transverse force
double nearfield(double dx, double dy) {
	double R = 0.5; // radius of embryo
	double lc = 1.0; // ratio between R and dc, the range for transverse force
	double dc = lc * R;

	double rij = sqrt(dx * dx + dy * dy);
	double dij = rij - 1.0;
	double nearf;

	if (dij < dc) {
		nearf = log(0.5 * lc / dij);
	}
	else {
		nearf = 0.0;
	}

	return nearf;
}

double drdt_xt(double dx, double dy, double wi, double wj) {
	//double f0 = 1.0; // coefficient for transverse force
	double nearf = nearfield(dx, dy);
	double rij = sqrt(dx * dx + dy * dy);
	double fnfx = 0.5 * (wi + wj) * f0 * nearf * dy / rij;

	return fnfx;
}

double drdt_yt(double dx, double dy, double wi, double wj) {
	//double f0 = 1.0; // coefficient for transverse force
	double nearf = nearfield(dx, dy);
	double rij = sqrt(dx * dx + dy * dy);
	double fnfy = -0.5 * (wi + wj) * f0 * nearf * dx / rij;

	return fnfy;
}

// function to get a vector of indices of neighbors
int* getNeighIdx(int i) {
	int* idxvec; // vector whose elements are the indices of the neighbors of particle i
	idxvec = (int*)malloc(6 * sizeof(int));

	int idx_row = i / numx; // index for the row
	int idx_col = i % numx; // index for the column
	int idx1; // index for neighbor 1
	int idx2;
	int idx3;
	int idx4;
	int idx5;
	int idx6;

	if (idx_row == 0) {
		idx3 = i + numx;
		idx5 = i - numx + N;

		if (idx_col == 0) {
			idx1 = i + 1;
			idx2 = i + numx - 1;
			idx4 = N - 1;
			idx6 = i + 2 * numx - 1;
		}
		else if (idx_col == numx - 1) {
			idx1 = i - numx + 1;
			idx2 = i - 1;
			idx4 = N - 2;
			idx6 = i + numx - 1;
		}
		else {
			idx1 = i + 1;
			idx2 = i - 1;
			idx4 = i - numx - 1 + N;
			idx6 = i + numx - 1;
		}
	}
	else if (idx_row == numy - 1) {
		idx4 = i - numx;
		idx6 = i + numx - N;

		if (idx_col == 0) {
			idx1 = i + 1;
			idx2 = N - 1;
			idx3 = 1;
			idx5 = i - numx + 1;
		}
		else if (idx_col == numx - 1) {
			idx1 = numx * (numy - 1);
			idx2 = i - 1;
			idx3 = 0;
			idx5 = numx * (numy - 2);
		}
		else {
			idx1 = i + 1;
			idx2 = i - 1;
			idx3 = i + numx + 1 - N;
			idx5 = i - numx + 1;
		}
	}
	else {
		if (idx_row % 2 == 0) { // even index of row
			idx3 = i + numx;
			idx5 = i - numx;

			if (idx_col == 0) {
				idx1 = i + 1;
				idx2 = i + numx - 1;
				idx4 = i - 1;
				idx6 = i + 2 * numx - 1;
			}
			else if (idx_col == numx - 1) {
				idx1 = i - numx + 1;
				idx2 = i - 1;
				idx4 = i - numx - 1;
				idx6 = i + numx - 1;
			}
			else {
				idx1 = i + 1;
				idx2 = i - 1;
				idx4 = i - numx - 1;
				idx6 = i + numx - 1;
			}
		}
		else { // odd index of row
			idx4 = i - numx;
			idx6 = i + numx;

			if (idx_col == 0) {
				idx1 = i + 1;
				idx2 = i + numx - 1;
				idx3 = i + numx + 1;
				idx5 = i - numx + 1;
			}
			else if (idx_col == numx - 1) {
				idx1 = i - numx + 1;
				idx2 = i - 1;
				idx3 = i + 1;
				idx5 = i - 2 * numx + 1;
			}
			else {
				idx1 = i + 1;
				idx2 = i - 1;
				idx3 = i + numx + 1;
				idx5 = i - numx + 1;
			}
		}
	}

	idxvec[0] = idx1;
	idxvec[1] = idx2;
	idxvec[2] = idx3;
	idxvec[3] = idx4;
	idxvec[4] = idx5;
	idxvec[5] = idx6;

	return idxvec;
}