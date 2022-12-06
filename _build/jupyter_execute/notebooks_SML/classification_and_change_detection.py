#!/usr/bin/env python
# coding: utf-8

# # Image classification and change detection
# A powerful application of remote sensing is detecting changes in land cover or land use. The natural environment changes from sudden and sometimes catastrophic events, but there is also a slower pace as in meandering rivers, or the annual cycle in vegetation. On longer time scales the environment changes due to gradual geomorphological changes and the effects of climete change. The built and managed environment also shows changes in land cover due to e.g. farming, dam building, city expansions, deforestation, or beach nourishment, to name a few.
# 
# Earth observation provides a fast and relatively cheap way to inventory and monitor these changes. A prerequisite for the successful application of remote sensing for change detection is that the changes in land cover result in changes in radiance values and that these radiance changes are large with respect to other causes of radiance changes such as changing atmospheric conditions, differences in soil moisture and differences in sun angle (Mas, 1999; Singh, 1989). Change detection methods can basically be subdivided into the following broad classes:
# 
# 1. **Image differencing:** registered images acquired at different times are subtracted to produce a residual image which represents the change between the two dates. Pixels of no radiance change are distributed around the mean while pixels of radiance change are distributed in the tails of the statistical distribution.
# 2. **Vegetation Index Differencing:** The NDVI is calculated (the normalized ratio of red and near infrared reflectance) for the images acquired at different dates. Next the NDVI values of the two images are subtracted and result in high positive or high negative values for change areas.
# 3. **Direct multi-date classification:** This method is based on the single analysis of a combined dataset of the two dates images aiming at identifying areas of changes. Two images are first merged into one multi-band image set. Classes where changes are occurring are expected to present statistics significantly different from where change did not take place and can so be identified. Unsupervised classification methods e.g. the isolate algorithm to reveal the spectral distance patterns in the data.
# 4. **Post-classification analysis:** The two (or more) images acquired at different dates are first independently classified using the same legend (number of classes and type of classes). Next, the two images are overlaid and compared by subtracting or otherwise. It should be noted that the accuracy of the change detection depends on the accuracies of each of the individual classification results and the geometric match of the imagery.