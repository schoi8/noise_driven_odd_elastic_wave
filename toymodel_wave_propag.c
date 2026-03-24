#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <fftw3.h>

#define numx 60 // number of columns. has to be even if for pbc.
#define numy 30 // number of rows. has to be even if for pbc.
#define N numx*numy // number of particles.

double getDist(double xi, double xj, double L);
double getPBCval(double x, double L);
int* getNeighIdx(int i);
int* getNeighIdx_fbc(int i);
int getNumNeigh(int i);

/*
	Toy model to show wave propagation with a selected mode. Use space-dependent multiplicative noise.
	Free boundary conditions have been used to make it more similar to the experimental setting.
	One iteration of the simulation so that one can generate the movie.
*/

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

	// simulation
	double dt = 0.01;
	int n_tot = 40000; // total number of time steps
	int n_interval = 40; // time step interval for data recording
	int n_rec = (int)(n_tot / n_interval);
	int n_eq = 10000;

	char basedir[] = "/directory_to_save_data/";
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

	FILE* fp_xarr = fopen(dir_xarr, "w");
	fclose(fp_xarr);
	FILE* fp_yarr = fopen(dir_yarr, "w");
	fclose(fp_yarr);
	FILE* fp_vxarr = fopen(dir_vxarr, "w");
	fclose(fp_vxarr);
	FILE* fp_vyarr = fopen(dir_vyarr, "w");
	fclose(fp_vyarr);
	
	char file_Cll[] = "Cll_toy";
	char dir_Cll[100];
	sprintf(dir_Cll, "%s%s%s", basedir, file_Cll, dat);
	char file_Cltr[] = "Cltr_toy";
	char dir_Cltr[100];
	sprintf(dir_Cltr, "%s%s%s", basedir, file_Cltr, dat);
	char file_Clti[] = "Clti_toy";
	char dir_Clti[100];
	sprintf(dir_Clti, "%s%s%s", basedir, file_Clti, dat);
	char file_Ctt[] = "Ctt_toy";
	char dir_Ctt[100];
	sprintf(dir_Ctt, "%s%s%s", basedir, file_Ctt, dat);

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.0; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing
	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.
	double D0 = 0.01;
	double sig = 1.0; // std of the Gaussian random variable
	double pi = acos(-1); // define pi
	double u0 = 0.05; // initial displacement amplitude
	double ka = 1.0;
	double k = 0.5;

	// miscellaneous indices
	int i, j, n;

	// perfect triangular lattice
	double* xlatt;
	xlatt = (double*)malloc(N * sizeof(double));
	double* ylatt;
	ylatt = (double*)malloc(N * sizeof(double));

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

	// current position
	double* xcur;
	xcur = (double*)malloc(N * sizeof(double));
	double* ycur;
	ycur = (double*)malloc(N * sizeof(double));

	// velocity of each particle
	double* vxcur;
	vxcur = (double*)malloc(N * sizeof(double));
	double* vycur;
	vycur = (double*)malloc(N * sizeof(double));

	// initial condition
	for (i = 0; i < N; i++) {
		xcur[i] = xlatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		ycur[i] = ylatt[i] + 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
	}

	FILE* fp_para = fopen(dir_para, "w");
	fprintf(fp_para, "numx = %i,  numy = %i, N = %i \n", numx, numy, N);
	fprintf(fp_para, "n_eq = %i, n_tot = %i, n_interval = %i, dt = %f \n", n_eq, n_tot, n_interval, dt);
	fprintf(fp_para, "tmax = n_tot*dt = %f, t_interval = n_interval*dt = %f \n", n_tot * dt, n_interval * dt);
	fprintf(fp_para, "R = %f, spacing = %f, r0 = %f \n", R, spacing, r0);
	fprintf(fp_para, "k = %f, ka = %f, u0 = %f \n", k, ka, u0);
	fprintf(fp_para, "Gaussian white noise with D0 = %f \n", D0);
	fclose(fp_para);
	
	// current matrix
	double dx = 0.4 * spacing;
	double dy = 0.4 * spacing;
	const int xnum = (int)round(Lx / dx);
	const int ynum = (int)round(Ly / dy);
	
	int wnum = (int)floor(n_rec / 2) + 1;

	double* jxmat;
	jxmat = (double*)calloc(xnum * ynum * n_rec, sizeof(double));
	double* jymat;
	jymat = (double*)calloc(xnum * ynum * n_rec, sizeof(double));
	
	int n_idx, ix, iy;
	double xij, yij, rij, nxij, nyij, xcur0, ycur0;
	int idx_neigh;
	int numNeigh;

	for (n = 0; n < n_eq + n_tot; n++) {
		for (i = 0; i < N; i++) {
			vxcur[i] = 0.0;
			vycur[i] = 0.0;

			int* neighvec = getNeighIdx_fbc(i);
			numNeigh = getNumNeigh(i);

			for (j = 0; j < numNeigh; j++) {
				idx_neigh = neighvec[j];
				xij = getDist(xcur[i], xcur[idx_neigh], Lx);
				yij = getDist(ycur[i], ycur[idx_neigh], Ly);
				rij = sqrt(xij * xij + yij * yij);
				nxij = xij / rij;
				nyij = yij / rij;

				vxcur[i] += -ka * (rij - r0) * nyij - k * (rij - r0) * nxij;
				vycur[i] += ka * (rij - r0) * nxij - k * (rij - r0) * nyij;
			}
			free(neighvec);
		}
		
		// update variables and save
		for (i = 0; i < N; i++) {
			xcur0 = xcur[i] + dt * vxcur[i] + sqrt(2 * D0 * dt * exp(-0.1 * xcur[i] * xcur[i]) * fabs(cos((0.5 * pi / r0) * xcur[i]))) * gsl_ran_gaussian(gr, sig);
			xcur[i] = xcur0; // free boundary condition
			
			ycur0 = ycur[i] + dt * vycur[i] + sqrt(2 * D0 * dt * exp(-0.1 * xcur[i] * xcur[i]) * fabs(cos((0.5 * pi / r0) * xcur[i]))) * gsl_ran_gaussian(gr, sig);
			ycur[i] = ycur0; // free boundary condition
		}

		if (n >= n_eq) {
			if ((n - n_eq) % n_interval == n_interval - 1) {
				fp_xarr = fopen(dir_xarr, "a");
				fp_yarr = fopen(dir_yarr, "a");
				fp_vxarr = fopen(dir_vxarr, "a");
				fp_vyarr = fopen(dir_vyarr, "a");

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
				}
				fclose(fp_xarr);
				fclose(fp_yarr);
				fclose(fp_vxarr);
				fclose(fp_vyarr);
				
				n_idx = (int)floor((n - n_eq) / n_interval);

				for (i = 0; i < N; i++) {
					ix = (int)floor(xcur[i] / dx);
					iy = (int)floor(ycur[i] / dy);

					jxmat[ix * ynum * n_rec + iy * n_rec + n_idx] += vxcur[i];
					jymat[ix * ynum * n_rec + iy * n_rec + n_idx] += vycur[i];
				}
			}
		}
	}
	free(xlatt);
	free(ylatt);
	free(xcur);
	free(ycur);
	free(vxcur);
	free(vycur);
	gsl_rng_free(gr); // free the random number generator

	fftw_complex* outjx;
	outjx = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjx;
	pjx = fftw_plan_dft_r2c_3d(xnum, ynum, n_rec, jxmat, outjx, FFTW_ESTIMATE);
	fftw_execute(pjx);
	fftw_destroy_plan(pjx);
	free(jxmat);

	fftw_complex* outjy;
	outjy = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * xnum * ynum * wnum);
	fftw_plan pjy;
	pjy = fftw_plan_dft_r2c_3d(xnum, ynum, n_rec, jymat, outjy, FFTW_ESTIMATE);
	fftw_execute(pjy);
	fftw_destroy_plan(pjy);
	free(jymat);

	double dkx = 2 * pi / Lx; // resolution in kx
	double dky = 2 * pi / Ly; // resolution in ky

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

	double* Cll; // current correlation Jl*Jl
	Cll = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_r; // current correlation Jl*Jt
	Clt_r = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Clt_i; // current correlation Jt*Jl
	Clt_i = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	double* Ctt; // current correlation Jt*Jt
	Ctt = (double*)calloc(xnum * ynum * wnum, sizeof(double));
	
	int ki_shift, kj_shift;
	double qxhat, qyhat, qx, qy, jxre, jxim, jyre, jyim;
	int ki, kj, ni;
	double jlre, jlim, jtre, jtim;

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

int* getNeighIdx_fbc(int i) {
	int* idxvec; // vector whose elements are the indices of the neighbors of particle i
	int idx_x = i % numx;
	int idx_y = i / numx;
	int numNeigh;

	if (idx_x == 0) {
		if (idx_y == 0) {
			numNeigh = 2;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x + 1 + idx_y * numx;
			idxvec[1] = idx_x + (idx_y + 1) * numx;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 3;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x + 1 + idx_y * numx;
			idxvec[1] = idx_x + (idx_y - 1) * numx;
			idxvec[2] = idx_x + 1 + (idx_y - 1) * numx;
		}
		else {
			if (idx_y % 2 == 0) {
				numNeigh = 5;
				idxvec = (int*)malloc(numNeigh * sizeof(int));
				idxvec[0] = idx_x + (idx_y - 1) * numx;
				idxvec[1] = idx_x + 1 + (idx_y - 1) * numx;
				idxvec[2] = idx_x + 1 + idx_y * numx;
				idxvec[3] = idx_x + (idx_y + 1) * numx;
				idxvec[4] = idx_x + 1 + (idx_y + 1) * numx;
			}
			else {
				numNeigh = 3;
				idxvec = (int*)malloc(numNeigh * sizeof(int));
				idxvec[0] = idx_x + (idx_y - 1) * numx;
				idxvec[1] = idx_x + 1 + idx_y * numx;
				idxvec[2] = idx_x + (idx_y + 1) * numx;
			}
		}
	}

	else if (idx_x == numx - 1) {
		if (idx_y == 0) {
			numNeigh = 3;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x - 1 + idx_y * numx;
			idxvec[1] = idx_x + (idx_y + 1) * numx;
			idxvec[2] = idx_x - 1 + (idx_y + 1) * numx;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 2;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x - 1 + idx_y * numx;
			idxvec[1] = idx_x + (idx_y - 1) * numx;
		}
		else {
			if (idx_y % 2 == 0) {
				numNeigh = 3;
				idxvec = (int*)malloc(numNeigh * sizeof(int));
				idxvec[0] = idx_x + (idx_y - 1) * numx;
				idxvec[1] = idx_x - 1 + idx_y * numx;
				idxvec[2] = idx_x + (idx_y + 1) * numx;
			}
			else {
				numNeigh = 5;
				idxvec = (int*)malloc(numNeigh * sizeof(int));
				idxvec[0] = idx_x + (idx_y - 1) * numx;
				idxvec[1] = idx_x - 1 + (idx_y - 1) * numx;
				idxvec[2] = idx_x - 1 + idx_y * numx;
				idxvec[3] = idx_x + (idx_y + 1) * numx;
				idxvec[4] = idx_x - 1 + (idx_y + 1) * numx;
			}
		}
	}

	else {
		if (idx_y == 0) {
			numNeigh = 4;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x - 1 + idx_y * numx;
			idxvec[1] = idx_x + 1 + idx_y * numx;
			idxvec[2] = idx_x - 1 + (idx_y + 1) * numx;
			idxvec[3] = idx_x + (idx_y + 1) * numx;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 4;
			idxvec = (int*)malloc(numNeigh * sizeof(int));
			idxvec[0] = idx_x - 1 + idx_y * numx;
			idxvec[1] = idx_x + 1 + idx_y * numx;
			idxvec[2] = idx_x + (idx_y - 1) * numx;
			idxvec[3] = idx_x + 1 + (idx_y - 1) * numx;
		}
		else {
			numNeigh = 6;
			idxvec = getNeighIdx(i); // array of neighbor indicess
		}
	}

	return idxvec;
}

int getNumNeigh(int i) {
	int idx_x = i % numx;
	int idx_y = i / numx;
	int numNeigh;

	if (idx_x == 0) {
		if (idx_y == 0) {
			numNeigh = 2;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 3;
		}
		else {
			if (idx_y % 2 == 0) {
				numNeigh = 5;
			}
			else {
				numNeigh = 3;
			}
		}
	}

	else if (idx_x == numx - 1) {
		if (idx_y == 0) {
			numNeigh = 3;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 2;
		}
		else {
			if (idx_y % 2 == 0) {
				numNeigh = 3;
			}
			else {
				numNeigh = 5;
			}
		}
	}

	else {
		if (idx_y == 0) {
			numNeigh = 4;
		}
		else if (idx_y == numy - 1) {
			numNeigh = 4;
		}
		else {
			numNeigh = 6;
		}
	}

	return numNeigh;
}