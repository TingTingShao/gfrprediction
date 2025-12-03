import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import scipy.stats as st
from statsmodels.stats.proportion import proportion_confint
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.stats.contingency_tables import mcnemar
import config
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
class Evaluation:
	evaluation_measures = ["r2","mae","mse", "imprecision", "median_bias"]

	def __init__(self,
				percentage = [0.1, 0.3],
				alpha = None
				):
		self.percentage = percentage
		self.alpha = alpha
	# def evaluate(self,
	# 			mGFR,
	# 			eGFR):
	# 	measure_columns = self.evaluation_measures[:]
	# 	results = [r2_score(mGFR, eGFR), 
	# 			mean_absolute_error(mGFR, eGFR),
	# 			mean_squared_error(mGFR, eGFR),
	# 			self.imprecision(mGFR, eGFR),
	# 			self.median_bias(mGFR, eGFR)
	# 			]
	# 	for p in self.percentage:
	# 		results.append(self.precision_percentage(mGFR, eGFR, p))	
	# 		measure_columns.append("precision_" + str(p))
	# 	results = pd.DataFrame(results)
	# 	results.insert(0, "metrics", measure_columns)
	# 	return results

	def evaluate(self, eGFR, mGFR):
		# Build a dictionary of metric names and their values
		metrics = {
			# "r2_score": r2_score(mGFR, eGFR),
			'ccc': self.lin(mGFR, eGFR),
			# "mean_absolute_error": mean_absolute_error(mGFR, eGFR),
			# "mean_squared_error": mean_squared_error(mGFR, eGFR),
			"median_bias": self.median_bias(mGFR, eGFR),
			"IQR": self.imprecision(mGFR, eGFR) #IQR
		}
		# Add the precision percentages for each percentage in self.percentage
		for p in self.percentage:
			metrics[f"precision_{p}"] = self.precision_percentage(mGFR, eGFR, p)
		
		# Return a Series where the index is the metric names
		return pd.Series(metrics, name="value")
			
	def precision_percentage(self,
						mGFR,
						eGFR,
						percentage
						):
		upper_bound = np.multiply(mGFR, 1.0 + percentage)
		lower_bound = np.multiply(mGFR, 1.0 - percentage)

		nb_instances = mGFR.shape[0]

		count = np.sum([v < upper_bound[i] and v > lower_bound[i] for i,v in enumerate(eGFR)])
		precision_percentage = (count/nb_instances).round(3)
		result = str(precision_percentage) + str(self._confidence_interval_precision(count, nb_instances)) 
		return result

	def imprecision(self,
					mGFR,
					eGFR,
						):
		measured_bias = eGFR - mGFR			
		first_quartile = np.quantile(measured_bias, 0.25).round(3)
		third_quartile = np.quantile(measured_bias, 0.75).round(3)
		# imprecision = np.std(measured_bias) 
		IQR = (third_quartile - first_quartile).round(3)
		result = str(IQR) + "(" + str(first_quartile) + "," + str(third_quartile) + ")"
		return result

		
	def median_bias(self,
			mGFR,
			eGFR):
		measured_bias = eGFR - mGFR
		# print(f"measured_bias is na: {measured_bias.isna().sum()}")
		median_bias = np.median(measured_bias).round(3) # median -> nanmedian ignore nan values
		# print("median_bias", median_bias)
		return str(median_bias) + str(self._confidence_interval(measured_bias))
 	

	def _fisher_transform(self, r):
		"""
		Fisher transformation: z = 0.5 * ln[(1+r)/(1-r)]
		Clipping r prevents division by zero if |r| equals 1.
		"""
		r = np.clip(r, -0.9999, 0.9999)
		return 0.5 * np.log((1 + r) / (1 - r))

	def _fisher_inverse(self, z):
		"""
		Inverse Fisher transformation: r = (exp(2*z)-1)/(exp(2*z)+1)
		"""
		return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

# translation from CCC calculation shared by Hans 
	def lin(self, y_true, y_pred):
		"""
		Replicates the VBA Lin function in Python.
		
		Steps:
		1. Remove missing/error entries.
		2. Compute averages for X (y_true) and Y (y_pred).
		3. Compute SXY, SX, and SY averaged over the valid data count.
		4. Compute the Pearson correlation r.
		5. Compute rc = 2 * SXY / [(AvgY - AvgX)^2 + SX + SY].
		6. Compute u2 = (AvgX - AvgY)^2 / sqrt(SX * SY).
		7. Compute rcc = Fisher(rc) via the Fisher transformation.
		8. Compute intermediate variables a, b, c.
		9. Compute SE = sqrt((a + b - c) / (j - 2)) [using j, the number of valid pairs].
		10. Compute the lower and upper limits on the Fisher scale (LLr, ULr),
			then transform them back (LL, UL).
		11. Return a formatted string "rc [LL, UL]" with 4-decimal precision.
		
		Parameters:
		y_true, y_pred: array-like sequences (lists, NumPy arrays, or pandas Series)
						containing the paired measurements.
		
		Returns:
		A string in the format "rc [LL, UL]".
		"""
		# Convert inputs to numpy arrays (floats) and filter out non-finite values.
		y_true = np.asarray(y_true, dtype=np.float64)
		y_pred = np.asarray(y_pred, dtype=np.float64)
		valid = np.isfinite(y_true) & np.isfinite(y_pred)
		X = y_true[valid]
		Y = y_pred[valid]
		
		j = len(X)
		if j < 3:
			raise ValueError("Not enough valid data points; need at least 3.")
		
		# Compute averages.
		avgX = np.mean(X)
		avgY = np.mean(Y)
		
		# Compute sums (averaged over j valid pairs).
		SXY = np.sum((X - avgX) * (Y - avgY)) / j
		SX  = np.sum((X - avgX) ** 2) / j
		SY  = np.sum((Y - avgY) ** 2) / j

		# Pearson correlation coefficient.
		r = np.corrcoef(X, Y)[0, 1]
		
		# Compute rc.
		denom = (avgY - avgX)**2 + SX + SY
		if denom == 0:
			raise ZeroDivisionError("Denominator is zero when computing rc.")
		rc = 2 * SXY / denom
		
		# Compute u2, guarding against division by zero.
		if SX * SY > 0:
			u2 = (avgX - avgY)**2 / np.sqrt(SX * SY)
		else:
			u2 = np.nan

		# Apply Fisher transformation to rc.
		rcc = self._fisher_transform(rc)
		
		# Compute intermediate quantities a, b, c.
		# Guard against division by zero.
		if (1 - rc**2) == 0 or r == 0 or r**2 == 0:
			a = b = c = np.nan
		else:
			a = (1 - r**2) * (rc**2) / ((1 - rc**2) * (r**2))
			b = 2 * (1 - rc) * (rc**3) * u2 / (((1 - rc**2)**2) * r)
			c = (rc**4) * (u2**2) / (2 * ((1 - rc**2)**2) * (r**2))
		
		# Compute the standard error using j-2 degrees of freedom.
		SE = np.sqrt((a + b - c) / (j - 2))
		
		# Confidence interval in Fisher's z-scale.
		LLr = rcc - 1.96 * SE
		ULr = rcc + 1.96 * SE
		
		# Convert back to the correlation scale.
		LL = self._fisher_inverse(LLr)
		UL = self._fisher_inverse(ULr)
		
		# Format the output.
		result_str = f"{rc:.4f} [{LL:.4f}, {UL:.4f}]"
		return result_str
	
	def _confidence_interval(self,
								data):
		n = len(data)
		q = 0.5
		lower_bound_index = np.ceil(n * q  - 1.96 * np.sqrt(n * q * (1 - q))).astype(int) - 1
		upper_bound_index = np.ceil(n * q  + 1.96 * np.sqrt(n * q * (1 - q))).astype(int) - 1

		lower_bound = np.sort(data)[lower_bound_index]
		upper_bound = np.sort(data)[upper_bound_index]
		confidence_interval = "(" + str(lower_bound.round(3)).replace("[","").replace("]","") + "," + str(upper_bound.round(3)).replace("[","").replace("]","") + ")"
		return confidence_interval

	def _confidence_interval_precision(self,
								n_success,
								samples):
		lower, upper = proportion_confint(n_success, samples, alpha=0.05, method='normal')
		lower = round(lower, 3)
		upper = round(upper, 3)
		return "(" + str(lower) + "," + str(upper) + ")"

	def mcnemar_external(self, data, methods, percentage):
		precision={}
		for method in methods:
			df=data[data['dataset']=='ext_val'].dropna()
			mGFR = df['mGFR'].values
			eGFR = df[method].values
			# percentage = percentage
			upper_bound = mGFR * (1 + percentage)
			lower_bound = mGFR * (1 - percentage)
			flags = (eGFR > lower_bound) & (eGFR < upper_bound)
			precision[method] = flags.tolist()   
		alpha = 0.05
		for A, B in combinations(methods, 2):
			flags_A = np.asarray(precision[A])
			flags_B = np.asarray(precision[B])
		
			n10 = np.sum(flags_A & ~flags_B)
			n01 = np.sum(~flags_A & flags_B)
			table = [
				[np.sum(flags_A & flags_B), n10],
				[    n01, np.sum(~flags_A & ~flags_B)]
			]
			res = mcnemar(table, exact=True)
			if res.pvalue < alpha:
				if n10 > n01:
					verdict = f"{A} is significantly better than {B}"
				else:
					verdict = f"{B} is significantly better than {A}"
			else:
				verdict = f"No significant difference between {A} and {B}"
		
			print(f"{A:10s} vs {B:10s}: p={res.pvalue:.4f} → {verdict}")
		
		

	def plot_bias(self, data, methods, path=None, format='eps'):
		"""
		Plot the bias between two datasets.
		
		Parameters:
		data1, data2: array-like sequences (lists, NumPy arrays, or pandas Series)
						containing the paired measurements.
		mGFR: array-like sequence containing the mGFR values.
		path: string, optional
			If provided, saves the plot to this path.
		"""
		data=data.dropna(subset=methods)
		mGFR=data[config.LABEL].values
		age=data['AGE'].values
		age = age.reshape(-1, 1)
		# bias=[]
		degree = 3
		template = Pipeline([
			('poly', PolynomialFeatures(degree, include_bias=True)),
			('quant', QuantileRegressor(quantile=0.5, alpha=0, solver='highs'))
		])
		age_range = np.linspace(age.min(), age.max(), 200).reshape(-1, 1)
		plt.figure(figsize=(8, 5))
		for i in methods:
			# bias.append(data[i].values - mGFR)
			bias= data[i].values - mGFR
			pipe= clone(template)
			pipe.fit(age, bias)
			y_pred= pipe.predict(age_range)
			# plotting
			plt.plot(age_range, y_pred, lw=2,
					label=f' {i} (deg={degree})')
		plt.xlabel('Age (years)')
		plt.ylabel('Median Bias (eGFR - mGFR)')
		plt.legend()
		plt.tight_layout()
		# plt.title('Mean P10 fit')		
		ax = plt.gca()
		ax.set_rasterized(True)
		if path is not None:
			plt.savefig(path, bbox_inches='tight', format=format, dpi=1200)
			plt.close()
		plt.show()
			
			
	def plot_p10(self, data, methods, path=None, format='eps'):
		"""
		Plot the bias between two datasets.
		
		Parameters:
		data1, data2: array-like sequences (lists, NumPy arrays, or pandas Series)
						containing the paired measurements.
		mGFR: array-like sequence containing the mGFR values.
		path: string, optional
			If provided, saves the plot to this path.
		"""
		data=data.dropna(subset=methods)
		# mGFR=data[config.LABEL].values
		age=data['AGE'].values
		age = age.reshape(-1, 1)
		# bias=[]
		degree = 3
		template = Pipeline([
			("spline", SplineTransformer(n_knots=3, degree=3, include_bias=False)),
			("lin", LinearRegression())
		])
		age_range = np.linspace(age.min(), age.max(), 200).reshape(-1, 1)
		plt.figure(figsize=(8, 5))
		for i in methods:
			# bias.append(data[i].values - mGFR)
			# bias= data[i].values - mGFR
			bias=(np.abs(data[i] - data["mGFR"]) / data["mGFR"] <= 0.10).astype(int)
			pipe= clone(template)
			pipe.fit(age, bias)
			y_pred= pipe.predict(age_range)
			# plotting
			plt.plot(age_range, y_pred, lw=2,
					label=f' {i} (deg={degree})')
			


		plt.xlabel('Age (years)')
		plt.ylabel('P10')
		plt.title('Mean P10 fit')
		plt.legend()
		plt.tight_layout()
		ax = plt.gca()
		ax.set_rasterized(True)
		if path is not None:
			plt.savefig(path, bbox_inches='tight', format=format, dpi=1200)
			plt.close()
		plt.show()


	def plot_p30(self, data, methods, path=None, format='eps'):
		"""
		Plot the bias between two methods.
		
		Parameters:
		data: array-like sequences (lists, NumPy arrays, or pandas Series)
						containing the paired measurements.
		methods: two column names to compare.
		path: string, optional
			If provided, saves the plot to this path.
		"""
		data=data.dropna(subset=methods)
		# mGFR=data[config.LABEL].values
		age=data['AGE'].values
		age = age.reshape(-1, 1)
		# bias=[]
		degree = 3
		template = Pipeline([
			("spline", SplineTransformer(n_knots=3, degree=3, include_bias=False)),
			("lin", LinearRegression())
		])
		age_range = np.linspace(age.min(), age.max(), 200).reshape(-1, 1)
		plt.figure(figsize=(8, 5))
		for i in methods:
			# bias.append(data[i].values - mGFR)
			# bias= data[i].values - mGFR
			bias=(np.abs(data[i] - data["mGFR"]) / data["mGFR"] <= 0.30).astype(int)
			pipe= clone(template)
			pipe.fit(age, bias)
			y_pred= pipe.predict(age_range)
			# plotting
			plt.plot(age_range, y_pred, lw=2,
					label=f' {i} (deg={degree})')
			


		plt.xlabel('Age (years)')
		plt.ylabel('P30')
		plt.title('Mean P30 fit')
		plt.legend()
		plt.tight_layout()
		ax = plt.gca()
		ax.set_rasterized(True)
		if path is not None:
			plt.savefig(path, bbox_inches='tight', format=format, dpi=1200)
			plt.close()
		plt.show()
