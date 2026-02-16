//
// Created by sammc on 2/15/26.
//
#pragma once
#ifndef CLANES_LEVEL1_H
#define CLANES_LEVEL1_H
#ifdef __cplusplus
#define RESTRICT
{
#else
#define RESTRICT restrict
#endif

void clanes_saxpy(int n,
                  float alpha,
                  const float* RESTRICT x,
                  int incx,
                  float* RESTRICT y,
                  int incy);
void clanes_daxpy(int n,
                  double alpha,
                  const double* RESTRICT x,
                  int incx,
                  double* RESTRICT y,
                  int incy);

void clanes_scopy(int n, const float* RESTRICT x, int incx, float* RESTRICT y, int incy);
void clanes_dcopy(int n, const double* RESTRICT x, int incx, double* RESTRICT y, int incy);

void clanes_sswap(int n, float* RESTRICT x, int incx, float* RESTRICT y, int incy);
void clanes_dswap(int n, double* RESTRICT x, int incx, double* RESTRICT y, int incy);

void clanes_sscal(int n, float alpha, float* RESTRICT x, int incx);
void clanes_dscal(int n, double alpha, double* RESTRICT x, int incx);

float clanes_sdot(int n, const float* RESTRICT x, int incx, const float* RESTRICT y, int incy);
double clanes_ddot(int n, const double* RESTRICT x, int incx, const double* RESTRICT y, int incy);

float clanes_snrm2(int n, const float* RESTRICT x, int incx);
double clanes_dnrm2(int n, const double* RESTRICT x, int incx);


float clanes_sasum(int n, const float* RESTRICT x, int incx);
double clanes_dasum(int n, const double* RESTRICT x, int incx);


int clanes_isamax(int n, const float* RESTRICT x, int incx);
int clanes_idamax(int n, const double* RESTRICT x, int incx);


void clanes_srot(int n,
                 float* RESTRICT x,
                 int incx,
                 float* RESTRICT y,
                 int incy,
                 float c,
                 float s);
void clanes_drot(int n,
                 double* RESTRICT x,
                 int incx,
                 double* RESTRICT y,
                 int incy,
                 double c,
                 double s);

void clanes_srotg(float* a, float* b, float* c, float* s);
void clanes_drotg(double* a, double* b, double* c, double* s);


void clanes_srotm(int n,
                  float* RESTRICT x,
                  int incx,
                  float* RESTRICT y,
                  int incy,
                  const float* param);
void clanes_drotm(int n,
                  double* RESTRICT x,
                  int incx,
                  double* RESTRICT y,
                  int incy,
                  const double* param);

void clanes_srotmg(float* d1, float* d2, float* b1, float b2, float* param);
void clanes_drotmg(double* d1, double* d2, double* b1, double b2, double* param);

#ifdef __cplusplus
}
#endif
#endif //CLANES_LEVEL1_H
