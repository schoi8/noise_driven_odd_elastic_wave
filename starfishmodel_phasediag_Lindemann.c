#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/*
	one realization
	get the dynamic Lindemann ratio to see if melting happens
	checks the overlap between the agents and reassigns the new location if there is overlap until there is no overlap.
	Since the code prints the value of dynamic Lindemann ratio while the simulation is running, if the Lindemann ratio is large enough and the simulation is stuck 
	because it cannot find a configuration without overlap, it should be manually aborted.
*/

#define numx 30 // number of columns. has to be even for pbc.
#define numy 30 // number of rows. has to be even for pbc.
#define f0 1.0 // amplitude of transverse force

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

	const int N = numx * numy;
	double sig = 1.0; // std for Gaussian randon variable

	// simulation time
	int n_tot = 1000000; // total number of time steps
	int n_interval = 1000; // time step interval for data recording
	int n_rec = (int)(n_tot / n_interval);
	double tcur = 0.0;
	double dt = 0.001;

	char basedir[] = "/directory_to_save_the_data/";
	char txt[] = ".txt";
	char dat[] = ".dat";

	char file_para[] = "simulation_para";
	char dir_para[100];
	sprintf(dir_para, "%s%s%s", basedir, file_para, txt);

	char file_Lratio[] = "Lratio";
	char dir_Lratio[100];
	sprintf(dir_Lratio, "%s%s%s", basedir, file_Lratio, txt);
	char file_Lratiocur[] = "Lratiocur";
	char dir_Lratiocur[100];
	sprintf(dir_Lratiocur, "%s%s%s", basedir, file_Lratiocur, txt);
	
	char file_xarr[] = "xarr";
	char dir_xarr[100];
	sprintf(dir_xarr, "%s%s%s", basedir, file_xarr, dat);
	char file_yarr[] = "yarr";
	char dir_yarr[100];
	sprintf(dir_yarr, "%s%s%s", basedir, file_yarr, dat);
	
	FILE* fp_Lratio = fopen(dir_Lratio, "w");
	fclose(fp_Lratio);
	FILE* fp_Lratiocur = fopen(dir_Lratiocur, "w");
	fclose(fp_Lratiocur);
	
	FILE* fp_xarr = fopen(dir_xarr, "w");
	fclose(fp_xarr);
	FILE* fp_yarr = fopen(dir_yarr, "w");
	fclose(fp_yarr);
	
	char file_xcur[] = "xcur";
	char dir_xcur[100];
	sprintf(dir_xcur, "%s%s%s", basedir, file_xcur, txt);
	char file_ycur[] = "ycur";
	char dir_ycur[100];
	sprintf(dir_ycur, "%s%s%s", basedir, file_ycur, txt);
	
	FILE* fp_xcur = fopen(dir_xcur, "w");
	fclose(fp_xcur);
	FILE* fp_ycur = fopen(dir_ycur, "w");
	fclose(fp_ycur);
	
	double pi = acos(-1); // define pi

	double R = 0.5; // embryo radius. rescaled to 2R = 1.
	double spacing = 1.2; // spacing scale.
	double r0 = 2 * R * spacing; // perfect crystal lattice spacing
	double Lx = numx * r0; // x length of the system box.
	double Ly = 0.5 * sqrt(3) * numy * r0; // y length of the system box.
	double rm = 3.8 * R; // neighbor detecting radius 
	
	double Fst0 = 53.7;
	double frep0 = 785.1;
	double w0 = 1.0;
	
	// rondom variable v0
	double vi; // "v0 value" of each particle i
	double v0 = 0.01; // average amplitude of self-propelling velocity
	double ve = 0.1; // noise amplitude of v0.

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
	double* uxinit;
	uxinit = (double*)malloc(N * sizeof(double));
	double* uyinit;
	uyinit = (double*)malloc(N * sizeof(double));

	for (i = 0; i < N; i++) {
		uxinit[i] = 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		uyinit[i] = 2 * u0 * ((double)rand() / (double)RAND_MAX - 0.5);
		xcur[i] = xlatt[i] + uxinit[i];
		ycur[i] = ylatt[i] + uyinit[i];
		thcur[i] = 2 * pi * (double)rand() / (double)RAND_MAX; // random angle from 0 to 2pi
	}
	
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

	double duij2 = 0.0; // difference between du of neighboring particles where du = u(t)-u(0) with u being displacement. Used to calculate the dynamic Lindemann ratio.
	double xi0t, yi0t, xj0t, yj0t; // displacement between t=0 and t=t
	double u2temp = 0.0; // average of displacement^2 to calculate the Lindemann ratio

	int endcode = 0;
	int checkOverlap = 0; // a variable to check whether the noise term causes an overlap event. If it does, reject it and pick a new noise term.
	int noOverlap = 0;

	for (n = 0; n < n_tot; n++) {
		u2temp = 0.0;

		for (i = 0; i < N; i++) {
			vxcur[i] = 0.0;
			vycur[i] = 0.0;

			int* neighvec = getNeighIdx(i); // array of neighbor indices
			duij2 = 0.0;
			xi0t = getDist(xcur[i], xlatt[i] + uxinit[i], Lx);
			yi0t = getDist(ycur[i], ylatt[i] + uyinit[i], Ly);

			for (j = 0; j < 6; j++) {
				idx_neigh = neighvec[j];
				xij = getDist(xcur[i], xcur[idx_neigh], Lx);
				yij = getDist(ycur[i], ycur[idx_neigh], Ly);
				rij = sqrt(xij * xij + yij * yij);

				xj0t = getDist(xcur[idx_neigh], xlatt[idx_neigh] + uxinit[idx_neigh], Lx);
				yj0t = getDist(ycur[idx_neigh], ylatt[idx_neigh] + uyinit[idx_neigh], Ly);
				duij2 += (xi0t - xj0t) * (xi0t - xj0t) + (yi0t - yj0t) * (yi0t - yj0t);

				if (rij < 2 * R) {
					printf("overlap happened at time %f which is time step %i \n", tcur, n);
					printf("overlap between %d and %d \n", i, idx_neigh);
					printf("x values are %f and %f \n", xcur[i], xcur[idx_neigh]);
					printf("y values are %f and %f \n", ycur[i], ycur[idx_neigh]);
					printf("calculated xdis %f \n", xij);
					printf("calculated ydis %f \n", yij);
					printf("Time taken: %.7fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
					endcode = 1;
					break;
				}
				else if (rij < rm) {
					vxcur[i] += drdt_xl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_xt(xij, yij, w0, w0);
					vycur[i] += drdt_yl(xij, yij, Fst0, frep0, Fst0, frep0) + drdt_yt(xij, yij, w0, w0);
				}
			}

			u2temp += duij2 / (6 * N * 2 * r0 * r0);

			free(neighvec);
			if (endcode == 1) {
				break;
			}
		}
		if (endcode == 1) {
			break;
		}

		if (n % (10 * n_interval) == 0) {
			printf("current time t = %f and dynamic Lindemann ratio = %f\n", tcur, u2temp);
		}

		if (n % n_interval == 0) {
			fp_Lratiocur = fopen(dir_Lratiocur, "w"); // in order to see the Lindemann ratio at the moment when things get stuck
			fprintf(fp_Lratiocur, "%f\n", u2temp);
			fclose(fp_Lratiocur);
		}
		else {
			fp_Lratiocur = fopen(dir_Lratiocur, "a"); // in order to see the Lindemann ratio at the moment when things get stuck
			fprintf(fp_Lratiocur, "%f\n", u2temp);
			fclose(fp_Lratiocur);
		}
		

		if (n % n_interval == n_interval - 1) { // I don't think I need to save the Lindemann parameter for every step...
			fp_Lratio = fopen(dir_Lratio, "a");
			fprintf(fp_Lratio, "%f\n", u2temp);
			fclose(fp_Lratio);
		}

		fp_xcur = fopen(dir_xcur, "w");
		fp_ycur = fopen(dir_ycur, "w");

		// update variables
		for (i = 0; i < N; i++) {
			checkOverlap = 0;
			int* neighvec = getNeighIdx(i);
			while (checkOverlap == 0) {
				noOverlap = 0;
				vi = ve * gsl_ran_gaussian(gr, sig) + v0;
				xcur0 = xcur[i] + dt * (vxcur[i] + vi * cos(thcur[i]));
				xcur0 = getPBCval(xcur0, Lx);
				ycur0 = ycur[i] + dt * (vycur[i] + vi * sin(thcur[i]));
				ycur0 = getPBCval(ycur0, Ly);

				for (j = 0; j < 6; j++) {
					idx_neigh = neighvec[j];
					xij = getDist(xcur0, xcur[idx_neigh], Lx);
					yij = getDist(ycur0, ycur[idx_neigh], Ly);
					rij = sqrt(xij * xij + yij * yij);

					if (rij > 2 * R) {
						noOverlap += 1;
					}
				}
				if (noOverlap == 6) {
					checkOverlap = 1;
				}
			}
			free(neighvec);

			vxnoise[i] = (vi - v0) * cos(thcur[i]);
			vynoise[i] = (vi - v0) * sin(thcur[i]);
			
			vxcur[i] += vi * cos(thcur[i]);
			vycur[i] += vi * sin(thcur[i]);

			xcur[i] = xcur0;
			ycur[i] = ycur0;

			thcur0 = thcur[i] + dt * w0;
			thcur[i] = getPBCval(thcur0, 2 * pi);

			// save this configuration
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
		}
		fclose(fp_xcur);
		fclose(fp_ycur);

		// save the movie for the later time
		if (n > n_tot - 200 * n_interval) {
			if (n % n_interval == n_interval - 1) {
				fp_xarr = fopen(dir_xarr, "a");
				fp_yarr = fopen(dir_yarr, "a");

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
				}
				fclose(fp_xarr);
				fclose(fp_yarr);
			}
		}
		tcur += dt;
	}
	free(xlatt);
	free(ylatt);
	free(uxinit);
	free(uyinit);
	free(xcur);
	free(ycur);
	free(vxcur);
	free(vycur);
	free(thcur);
	gsl_rng_free(gr); // free the random number generator
	free(vxnoise);
	free(vynoise);

	if (endcode == 1) {
		exit(0);
	}
	
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
	const int N = numx * numy;
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