# coding:utf8

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def zipf_mandelbrot_pmf(k, s=1.2, q=1.0, K=100):
	# k: array-like of integers (>=1)
	ks = np.array(k, dtype=np.int64)
	denom = np.sum((np.arange(1, K+1) + q) ** (-s))
	pmf = (ks + q) ** (-s) / denom
	# For k > K, probability ~0
	pmf[ks > K] = 0.0
	return pmf


# --- Fonctions statistiques (moyenne et écart-type) ---
def stats_dirac(k0):
	return float(k0), 0.0


def stats_uniform_discrete(a, b):
	n = b - a + 1
	mean = 0.5 * (a + b)
	var = ((n ** 2) - 1) / 12.0
	return float(mean), math.sqrt(var)


def stats_binomial(n, p):
	mean = n * p
	var = n * p * (1 - p)
	return float(mean), math.sqrt(var)


def stats_poisson(lam):
	mean = float(lam)
	return mean, math.sqrt(mean)


def stats_zipf_mandelbrot(s=1.2, q=1.0, K=100):
	ks = np.arange(1, K + 1)
	pmf = zipf_mandelbrot_pmf(ks, s=s, q=q, K=K)
	mean = float(np.sum(ks * pmf))
	second = float(np.sum((ks ** 2) * pmf))
	var = max(0.0, second - mean ** 2)
	return mean, math.sqrt(var)


def stats_poisson_samples(lam, size=2000):
	samples = stats.poisson.rvs(mu=lam, size=size)
	return float(np.mean(samples)), float(np.std(samples, ddof=0))


def stats_normal(mu, sigma):
	return float(mu), float(sigma)


def stats_lognormal(s, scale=1.0):
	dist = stats.lognorm(s, scale=scale)
	return float(dist.mean()), float(dist.std())


def stats_uniform_cont(low, high):
	dist = stats.uniform(loc=low, scale=(high - low))
	return float(dist.mean()), float(dist.std())


def stats_chi2(df):
	dist = stats.chi2(df)
	return float(dist.mean()), float(dist.std())


def stats_pareto(b):
	dist = stats.pareto(b)
	return float(dist.mean()), float(dist.std())


def compute_and_save_stats(path='plots/stats.csv'):
	import csv

	rows = []
	# Discrètes
	rows.append(('Dirac(k=5)',) + stats_dirac(5))
	rows.append(('Uniforme discrète [0,10]',) + stats_uniform_discrete(0, 10))
	rows.append(('Binomiale(n=20,p=0.3)',) + stats_binomial(20, 0.3))
	rows.append(('Poisson(λ=4)',) + stats_poisson(4.0))
	mz_mean, mz_std = stats_zipf_mandelbrot(s=1.2, q=1.0, K=60)
	rows.append(('Zipf-Mandelbrot(s=1.2,q=1.0,K=60)', mz_mean, mz_std))

	# Continues
	rows.append(('Poisson(KDE samples λ=4)',) + stats_poisson_samples(4.0, size=2000))
	rows.append(('Normale(0,1)',) + stats_normal(0.0, 1.0))
	rows.append(('Log-Normale(s=0.6)',) + stats_lognormal(0.6))
	rows.append((f'Uniforme continue [{-2.0},{3.0}]',) + stats_uniform_cont(-2.0, 3.0))
	rows.append((f'Chi2(df=3)',) + stats_chi2(3))
	rows.append((f'Pareto(b=3.0)',) + stats_pareto(3.0))

	ensure_dir(os.path.dirname(path) or '.')
	with open(path, 'w', newline='', encoding='utf8') as f:
		writer = csv.writer(f)
		writer.writerow(['distribution', 'mean', 'std'])
		for r in rows:
			writer.writerow(r)

	return path


def plot_discrete_pmf(name, ks, pmf, ax):
	ax.bar(ks, pmf, align='center', alpha=0.7)
	ax.set_title(name)
	ax.set_xlabel('k')
	ax.set_ylabel('P(X=k)')


def plot_continuous_pdf(name, xs, pdf, ax):
	ax.plot(xs, pdf, lw=2)
	ax.set_title(name)
	ax.set_xlabel('x')
	ax.set_ylabel('densité')


def ensure_dir(d):
	if not os.path.exists(d):
		os.makedirs(d)


def main():
	ensure_dir('plots')

	# --- Discrètes ---
	fig, axes = plt.subplots(2, 3, figsize=(12, 8))
	axes = axes.ravel()

	# 1) Loi de Dirac (masse en k0)
	k0 = 5
	ks = np.arange(0, 16)
	pmf_dirac = np.zeros_like(ks, dtype=float)
	pmf_dirac[ks == k0] = 1.0
	plot_discrete_pmf('Dirac (masse en {})'.format(k0), ks, pmf_dirac, axes[0])

	# 2) Loi uniforme discrète sur [a,b]
	a, b = 0, 10
	ks = np.arange(a, b+1)
	pmf_unif = np.ones_like(ks, dtype=float) / len(ks)
	plot_discrete_pmf('Uniforme discrète [{}..{}]'.format(a, b), ks, pmf_unif, axes[1])

	# 3) Binomiale
	n, p = 20, 0.3
	ks = np.arange(0, n+1)
	pmf_binom = stats.binom.pmf(ks, n, p)
	plot_discrete_pmf('Binomiale(n={}, p={})'.format(n, p), ks, pmf_binom, axes[2])

	# 4) Poisson (discrète)
	lam = 4.0
	ks = np.arange(0, 21)
	pmf_pois = stats.poisson.pmf(ks, lam)
	plot_discrete_pmf('Poisson(λ={})'.format(lam), ks, pmf_pois, axes[3])

	# 5) Zipf-Mandelbrot (implémentation finie)
	K = 60
	ks = np.arange(1, K+1)
	pmf_zm = zipf_mandelbrot_pmf(ks, s=1.2, q=1.0, K=K)
	plot_discrete_pmf('Zipf-Mandelbrot (s=1.2, q=1.0)', ks, pmf_zm, axes[4])

	# empty placeholder or repeat
	axes[5].axis('off')

	plt.tight_layout()
	fig.savefig(os.path.join('plots', 'discrete_distributions.png'))

	# --- Continues ---
	fig2, axes2 = plt.subplots(3, 2, figsize=(12, 12))
	axes2 = axes2.ravel()

	# NOTE: User listed Poisson in continues — we'll show a smoothed KDE of Poisson samples
	# 1) Poisson (échantillons + KDE pour approcher une densité continue)
	lam = 4.0
	samples = stats.poisson.rvs(mu=lam, size=2000)
	xs = np.linspace(0, samples.max()+3, 200)
	# KDE on continuous grid
	kde = stats.gaussian_kde(samples)
	pdf_kde = kde(xs)
	plot_continuous_pdf('Poisson (KDE des échantillons λ={})'.format(lam), xs, pdf_kde, axes2[0])

	# 2) Normale
	mu, sigma = 0.0, 1.0
	xs = np.linspace(mu-4*sigma, mu+4*sigma, 400)
	pdf_norm = stats.norm.pdf(xs, loc=mu, scale=sigma)
	plot_continuous_pdf('Normale(μ={}, σ={})'.format(mu, sigma), xs, pdf_norm, axes2[1])

	# 3) Log-normale
	s = 0.6  # shape parameter
	xs = np.linspace(1e-3, 8, 400)
	pdf_lognorm = stats.lognorm.pdf(xs, s)
	plot_continuous_pdf('Log-Normale(s={})'.format(s), xs, pdf_lognorm, axes2[2])

	# 4) Uniforme continue
	low, high = -2.0, 3.0
	xs = np.linspace(low-0.5, high+0.5, 400)
	pdf_unif = stats.uniform.pdf(xs, loc=low, scale=(high-low))
	plot_continuous_pdf('Uniforme continue [{}, {}]'.format(low, high), xs, pdf_unif, axes2[3])

	# 5) Loi du χ²
	df = 3
	xs = np.linspace(0, 12, 400)
	pdf_chi2 = stats.chi2.pdf(xs, df)
	plot_continuous_pdf('Chi2(df={})'.format(df), xs, pdf_chi2, axes2[4])

	# 6) Pareto (scipy pareto uses >1 support)
	b = 3.0  # shape
	xs = np.linspace(1e-3, 6, 400)
	pdf_pareto = stats.pareto.pdf(xs, b)
	plot_continuous_pdf('Pareto(b={})'.format(b), xs, pdf_pareto, axes2[5])

	plt.tight_layout()
	fig2.savefig(os.path.join('plots', 'continuous_distributions.png'))

	print('Graphiques sauvegardés dans le dossier plots/')
	stats_path = compute_and_save_stats('plots/stats.csv')
	print(f'Statistiques sauvegardées dans {stats_path}')


if __name__ == '__main__':
	main()

