import numpy
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.ticker as ticker
import warnings
from sklearn.linear_model import QuantileRegressor


from ._rangeFrameLocator import rangeFrameLocator, rangeFrameLabler
from ._detrend import detrend as detrendFun
from ._calculateConfidenceIntervals import calculateConfidenceIntervals


def blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=95, confidenceIntervalMethod='approximate', percentage=False, precision=False, detrend=None, title=None, ax=None, figureSize=(10,7), dpi=72, savePath=None, figureFormat='png', meanColour='#6495ED', loaColour='coral', pointColour='#6495ED'):
	"""
	blandAltman(data1, data2, limitOfAgreement=1.96, confidenceInterval=None, **kwargs)

	Generate a Bland-Altman [#]_ [#]_ plot to compare two sets of measurements of the same value.

	Confidence intervals on the limit of agreement may be calculated using:
	- 'exact paired' uses the exact paired method described by Carkeet [#]_
	- 'approximate' uses the approximate method described by Bland & Altman

	The exact paired method will give more accurate results when the number of paired measurements is low (approx < 100), at the expense of much slower plotting time.

	The *detrend* option supports the following options:
	- ``None`` do not attempt to detrend data - plots raw values
	- 'Linear' attempt to model and remove a multiplicative offset between each assay by linear regression
	- 'ODR' attempt to model and remove a multiplicative offset between each assay by Orthogonal distance regression

	:param data1: List of values from the first method
	:type data1: list like
	:param data2: List of paired values from the second method
	:type data2: list like
	:param float limitOfAgreement: Multiples of the standard deviation to plot limit of agreement bounds at (defaults to 1.96)
	:param confidenceInterval: If not ``None``, plot the specified percentage confidence interval on the mean and limits of agreement
	:param str confidenceIntervalMethod: Method used to calculated confidence interval on the limits of agreement
	:type confidenceInterval: None or float
	:param detrend: If not ``None`` attempt to detrend by the method specified
	:type detrend: None or str
	:param bool percentage: If ``True``, plot differences as percentages (instead of in the units the data sources are in)
	:param str title: Title text for the figure
	:param matplotlib.axes._subplots.AxesSubplot ax: Matplotlib axis handle - if not `None` draw into this axis rather than creating a new figure
	:param figureSize: Figure size as a tuple of (width, height) in inches
	:type figureSize: (float, float)
	:param int dpi: Figure resolution
	:param str savePath: If not ``None``, save figure at this path
	:param str figureFormat: When saving figure use this format
	:param str meanColour: Colour to use for plotting the mean difference
	:param str loaColour: Colour to use for plotting the limits of agreement
	:param str pointColour: Colour for plotting data points

	.. [#] Altman, D. G., and Bland, J. M. “Measurement in Medicine: The Analysis of Method Comparison Studies” Journal of the Royal Statistical Society. Series D (The Statistician), vol. 32, no. 3, 1983, pp. 307–317. `JSTOR <https://www.jstor.org/stable/2987937>`_.
	.. [#] Altman, D. G., and Bland, J. M. “Measuring agreement in method comparison studies” Statistical Methods in Medical Research, vol. 8, no. 2, 1999, pp. 135–160. `DOI <https://doi.org/10.1177/096228029900800204>`_.
	.. [#] Carkeet, A. "Exact Parametric Confidence Intervals for Bland-Altman Limits of Agreement" Optometry and Vision Science, vol. 92, no 3, 2015, pp. e71–e80 `DOI <https://doi.org/10.1097/OPX.0000000000000513>`_.
	"""
	if not limitOfAgreement > 0:
		raise ValueError('"limitOfAgreement" must be a number greater than zero.') 

	# Try to coerce variables to numpy arrays
	data1 = numpy.asarray(data1)
	data2 = numpy.asarray(data2)

	data2, slope, slopeErr = detrendFun(detrend, data1, data2)

	mean = numpy.mean([data1, data2], axis=0)

	if percentage==True:
		diff = ((data1 - data2) / mean) * 100
	if precision==True:
		diff =((data1-data2)/data2)*100
	elif percentage==False & precision==False:
		diff = data1 - data2

	md = numpy.mean(diff)

	sd = numpy.std(diff, axis=0)

	if confidenceInterval:
		confidenceIntervals = calculateConfidenceIntervals(md, sd, len(diff), limitOfAgreement, confidenceInterval, confidenceIntervalMethod)

	else:
		confidenceIntervals = dict()

	ax = _drawBlandAltman(mean, diff, md, sd, percentage,precision,
						  limitOfAgreement,
						  confidenceIntervals,
						  (detrend, slope, slopeErr),
						  title,
						  ax,
						  figureSize,
						  dpi,
						  savePath,
						  figureFormat,
						  meanColour,
						  loaColour,
						  pointColour)

	if ax is not None:
		return ax


def _drawBlandAltman(mean, diff, md, sd, percentage, precision, limitOfAgreement, confidenceIntervals, detrend, title, ax, figureSize, dpi, savePath, figureFormat, meanColour, loaColour, pointColour):
	"""
	Sub function to draw the plot.
	"""
	if ax is None:
		fig, ax = plt.subplots(figsize=figureSize, dpi=dpi)
		draw = True
	else:
		draw = False
	##
	# Plot CIs if calculated
	##
	# if 'mean' in confidenceIntervals.keys():
	# 	ax.axhspan(confidenceIntervals['mean'][0],
	# 				confidenceIntervals['mean'][1],
	# 				facecolor=meanColour, alpha=0.2)

	# if 'upperLoA' in confidenceIntervals.keys():
	# 	ax.axhspan(confidenceIntervals['upperLoA'][0],
	# 				confidenceIntervals['upperLoA'][1],
	# 				facecolor=loaColour, alpha=0.2)

	# if 'lowerLoA' in confidenceIntervals.keys():
	# 	ax.axhspan(confidenceIntervals['lowerLoA'][0],
	# 				confidenceIntervals['lowerLoA'][1],
	# 				facecolor=loaColour, alpha=0.2)

	##
	# Plot the median diff and LoA
	##

	if precision==False:
		md = numpy.median(diff)
		first_quartile = numpy.quantile(diff, 0.25).round(3)
		third_quartile = numpy.quantile(diff, 0.75).round(3)
		ax.axhline(md, color=meanColour, linestyle='--')
		ax.axhline(third_quartile, color=loaColour, linestyle='--')
		ax.axhline(first_quartile, color=loaColour, linestyle='--')
	elif precision==True:
		md=0
		precision_01=10
		precision_03=30
		ax.axhline(md, color=meanColour, linestyle='--')
		ax.axhline(precision_01, color=loaColour, linestyle='--')
		ax.axhline(-precision_01, color=loaColour, linestyle='--')
		ax.axhline(precision_03, color=loaColour, linestyle='--')
		ax.axhline(-precision_03, color=loaColour, linestyle='--')

	# quantiles=[0.05, 0.5, 0.95]
	# predictions={}
	# out_bounds_predictions=numpy.zeros_like(mean)
	# for quantile in quantiles:
	# 	qr=QuantileRegressor(quantile=quantile, alpha=0)
	# 	y_pred=qr.fit(mean.reshape(-1, 1), diff).predict(mean.reshape(-1, 1))
	# 	predictions[quantile]=y_pred
	# 	if quantile==min(quantiles):
	# 		out_bounds_predictions=numpy.logical_or(
	# 			out_bounds_predictions, y_pred >= diff
	# 		)
	# 	elif quantile==max(quantiles):
	# 		out_bounds_predictions=numpy.logical_or(
	# 			out_bounds_predictions, y_pred <= diff
	# 		)
	# for quantile, y_pred in predictions.items():
	# 	plt.plot(mean, diff, label=f"Quantile: {quantile}")

	# plt.scatter(
	# 	mean[out_bounds_predictions],
	# 	diff[out_bounds_predictions],
	# 	color='black',
	# 	marker='+',
	# 	alpha=0.5,
	# 	label='Outisde interval'
	# )
	# plt.scatter(
	# 	mean[~out_bounds_predictions],
	# 	diff[~out_bounds_predictions],
	# 	color='black',
	# 	alpha=0.5,
	# 	label='Inside interval'
	# )
	qr=QuantileRegressor(quantile=0.5, alpha=0)
	y_pred=qr.fit(mean.reshape(-1, 1), diff).predict(mean.reshape(-1, 1))
	ax.plot(mean, y_pred, label=f"Quantile: {0.5}", color='black', lw=2)
	slope = qr.coef_[0]
	intercept = qr.intercept_
	# print(qr.intercept_, qr.coef_)
	# slope, intercept = numpy.polyfit(mean, diff, 1)
	# ax.axline(xy1=(0, intercept), slope=slope, color='black', lw=2)
	stats = (f'Slope = {slope:.2f}\n'
			f'Intercept = {intercept:.2f}')
	bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
	ax.text(0.95, 0.07, stats, fontsize=9, bbox=bbox,
			transform=ax.transAxes, horizontalalignment='right')
	# degree=2
	# coef = numpy.polyfit(mean, diff, deg=degree)
	# poly = numpy.poly1d(coef)
	# x_fit = numpy.linspace(mean.min(), mean.max(), 200)
	# plt.plot(x_fit, poly(x_fit), lw=2, label=f'Polynomial fit (deg={degree})')

	##
	# Plot the data points
	##
	ax.scatter(mean, diff, alpha=0.5, c=pointColour)

	trans = transforms.blended_transform_factory(
		ax.transAxes, ax.transData)

	limitOfAgreementRange = (md + (limitOfAgreement * sd)) - (md - limitOfAgreement*sd)
	offset = (limitOfAgreementRange / 100.0) * 1.5

	if precision==False:
		ax.text(0.98, md + offset, 'Median', ha="right", va="bottom", transform=trans)
		ax.text(0.98, md - offset, f'{md:.2f}', ha="right", va="top", transform=trans)
		ax.text(0.98, first_quartile, f'1st quantile {first_quartile:.2f}', ha="right", va="top", transform=trans)
		ax.text(0.98, third_quartile, f'3rd quantile {third_quartile:.2f}', ha="right", va="bottom", transform=trans)
	else: 
		ax.text(0.98, 0, '0', ha="right", va="bottom", transform=trans)
		ax.text(0.98, 10, '10', ha="right", va="top", transform=trans)
		ax.text(0.98, -10, '-10', ha="right", va="bottom", transform=trans)
		ax.text(0.98, 30, '30', ha="right", va="top", transform=trans)
		ax.text(0.98, -30, '-30', ha="right", va="bottom", transform=trans)

	# ax.text(0.98, md + (limitOfAgreement * sd) + offset, f'+{limitOfAgreement:.2f} SD', ha="right", va="bottom", transform=trans)
	# ax.text(0.98, md + (limitOfAgreement * sd) - offset, f'{md + limitOfAgreement*sd:.2f}', ha="right", va="top", transform=trans)

	# ax.text(0.98, md - (limitOfAgreement * sd) - offset, f'-{limitOfAgreement:.2f} SD', ha="right", va="top", transform=trans)
	# ax.text(0.98, md - (limitOfAgreement * sd) + offset, f'{md - limitOfAgreement*sd:.2f}', ha="right", va="bottom", transform=trans)

	# Only draw spine between extent of the data
	ax.spines['left'].set_bounds(min(diff), max(diff))
	ax.spines['bottom'].set_bounds(min(mean), max(mean))

	# Hide the right and top spines
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

	if precision==True:
		ax.set_ylabel('(Method - mGFR)/mGFR (%)')
	if percentage==True:
		ax.set_ylabel('(Method - mGFR)/mean of methods (%)')
	elif precision==False & percentage==False:
		ax.set_ylabel('(Method - mGFR) mL/min/1.73m2')
	ax.set_xlabel('(Method + mGFR)/2 mL/min/1.73m2')

	tickLocs = ax.xaxis.get_ticklocs()
	cadenceX = tickLocs[2] - tickLocs[1]
	tickLocs = rangeFrameLocator(tickLocs, (min(mean), max(mean)))
	ax.xaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	tickLocs = ax.yaxis.get_ticklocs()
	cadenceY = tickLocs[2] - tickLocs[1]
	tickLocs = rangeFrameLocator(tickLocs, (min(diff), max(diff)))
	ax.yaxis.set_major_locator(ticker.FixedLocator(tickLocs))

	ax.figure.canvas.draw() # Force drawing to populate tick labels

	labels = rangeFrameLabler(ax.xaxis.get_ticklocs(), [item.get_text() for item in ax.get_xticklabels()], cadenceX)
	ax.set_xticklabels(labels)

	labels = rangeFrameLabler(ax.yaxis.get_ticklocs(), [item.get_text() for item in ax.get_yticklabels()], cadenceY)
	ax.set_yticklabels(labels)


	ax.patch.set_alpha(0)

	if detrend[0] is None:
		pass
	else:
		plt.text(1, -0.1, f'{detrend[0]} slope correction factor: {detrend[1]:.2f} ± {detrend[2]:.2f}', ha='right', transform=ax.transAxes)

	if title:
		ax.set_title(title)

	##
	# Save or draw
	##
	# ax.set_rasterized(True)
	if (savePath is not None) & draw:
		
		fig.savefig(savePath, format=figureFormat)
		plt.close()
	elif draw:
		plt.show()
	else:
		return ax
