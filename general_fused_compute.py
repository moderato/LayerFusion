from __future__ import absolute_import as _abs
import tvm

from topi.nn.dilate import dilate
from topi.nn.pad import pad
from topi.nn.util import get_pad_tuple
from topi.util import simplify

from helper import *

def fused_convs(input_data, filters, resnet_block=False):
	out_dtype = input_data.dtype

	Input = None
	stages = [[input_data]]
	params = [[input_data]]

	for f in filters:
		Input = stages[-1][-1]
		Filter = f.placeholder
		layout = f.layout
		depthwise = f.depthwise
		bn_relu = f.bn_relu
		kernel = f.kernel
		stride = f.stride
		padding = f.padding
		dilation = f.dilation
		tmp_stages = []
		tmp_params = []

		assert not (depthwise and kernel == 1) # Don't consider 1by1 depthwise

		padded_count = 0
		conv_count = 0
		depthwise_count = 0

		if isinstance(stride, int):
			stride_h = stride_w = stride
		else:
			stride_h, stride_w = stride

		if isinstance(dilation, int):
			dilation_h = dilation_w = dilation
		else:
			dilation_h, dilation_w = dilation

		batch, in_height, in_width, in_channel = Input.shape
		kernel_h, kernel_w, kernel_channel, tmp = Filter.shape
		if depthwise:
			channel_multiplier = tmp
		else:
			num_filter = tmp

		# compute the output shape
		dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
		dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
		pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
			padding, (dilated_kernel_h, dilated_kernel_w))

		out_channel = simplify(in_channel * channel_multiplier) if depthwise else num_filter
		out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
		out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

		if f.kernel > 1:
			# print("Padding is needed!")

			pad_before = [0, pad_top, pad_left, 0]
			pad_after = [0, pad_down, pad_right, 0]

			PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput_{}".format(padded_count))
			padded_count += 1
			stages.append([PaddedInput])

			# Update Input
			Input = PaddedInput
			batch, in_height, in_width, in_channel = Input.shape

		if not depthwise:
			rc = tvm.reduce_axis((0, in_channel), name='rc')
		if kernel > 1:
			ry = tvm.reduce_axis((0, kernel_h), name='ry')
			rx = tvm.reduce_axis((0, kernel_w), name='rx')

		if not depthwise: # Normal convolution
			if kernel > 1:
				Output = tvm.compute(
				(batch, out_height, out_width, out_channel),
				lambda nn, yy, xx, ff: tvm.sum(
					Input[nn, yy * stride_h + ry * dilation_h,
								xx * stride_w + rx * dilation_w, rc].astype(out_dtype) *
					Filter[ry, rx, rc, ff].astype(out_dtype), axis=[ry, rx, rc]),
					name="Conv2dOutput_{}".format(conv_count), tag="conv2d_nhwc")
			else: # Only reduce rc axis
				Output = tvm.compute(
				(batch, out_height, out_width, out_channel),
				lambda nn, yy, xx, ff: tvm.sum(
					Input[nn, yy * stride_h, xx * stride_w, rc].astype(out_dtype) *
					Filter[0, 0, rc, ff].astype(out_dtype), axis=[rc]),
					name="Conv2dOutput_{}".format(conv_count), tag="conv2d_nhwc")
			conv_count += 1
		else: # Depthwise convolution (kernel > 1)
			Output = tvm.compute(
			(batch, out_height, out_width, out_channel),
			lambda b, i, j, c: tvm.sum(
				(Input[b, i*stride_h + ry*dilation_h, j*stride_w + rx*dilation_w,
							 tvm.indexdiv(c, channel_multiplier)].astype(out_dtype) *
				Filter[ry, rx, tvm.indexdiv(c, channel_multiplier), tvm.indexmod(c, channel_multiplier)].astype(out_dtype)),
				axis=[ry, rx]),
			name='DepthwiseConv2dOutput_{}'.format(depthwise_count), tag="depthwise_nhwc")
			depthwise_count += 1

		tmp_stages.append(Output)
		tmp_params.append(Filter)

		if bn_relu is not None:
			_, _, _, out_channel = Output.shape
			tensor_name = Output.name
			number = tensor_name.split('_')[-1]
			Scale = tvm.placeholder((out_channel),
								name='Scale_{}_{}'.format(
									'DepthwiseConv2d' if depthwise else 'Conv2d', number))
			Shift = tvm.placeholder((out_channel),
								name='Shift_{}_{}'.format(
									'DepthwiseConv2d' if depthwise else 'Conv2d', number))
			ScaleShift =  tvm.compute(Output.shape, lambda b, i, j, c: Output[b, i, j, c] * Scale[c] + Shift[c],
								name='ScaleShift_{}_{}'.format(
									'DepthwiseConv2d' if depthwise else 'Conv2d', number),
								tag='scaleshift_nhwc')
			if bn_relu == 'relu':
				ReLU = tvm.compute(ScaleShift.shape, lambda *i: tvm.max(ScaleShift(*i), tvm.const(0, ScaleShift.dtype)),
								name='ReLU_{}_{}'.format(
									'DepthwiseConv2d' if depthwise else 'Conv2d', number),
								tag='relu_nhwc')
			else: # 'relu6'
				ReLU = tvm.compute(ScaleShift.shape, lambda *i: tvm.min(
									tvm.max(ScaleShift(*i), tvm.const(0, ScaleShift.dtype)),
									tvm.const(6, ScaleShift.dtype)),
								name='ReLU6_{}_{}'.format(
									'DepthwiseConv2d' if depthwise else 'Conv2d', number),
								tag='relu')

			tmp_stages.append(ScaleShift)
			tmp_stages.append(ReLU)
			tmp_params.append(Scale)
			tmp_params.append(Shift)

		stages.append(tmp_stages)
		params.append(tmp_params)

	if resnet_block:
		First = stages[0][0]
		Last = stages[-1][-1]
		assert (First.shape == Last.shape)
		Output = tvm.compute(
			(batch, out_height, out_width, out_channel),
			lambda b, i, j, c: tvm.sum(
				(First[b, i, j, c].astype(out_dtype) + 
				(Last[b, i, j, c]).astype(out_dtype))),
			name='ElementwiseAddOutput_{}'.format(depthwise_count), tag="elem_nhwc")
		stages.append([Output])

	params.append([stages[-1][-1]]) # Final output

	return stages, params

if __name__ == "__main__":
	Input = tvm.placeholder((1, 56, 56, 128), name='Input')

	Filters = []
	Filters.append(FilterParams(
					tvm.placeholder((3, 3, 128, 1), name='DepthwiseFilter_0'),
					depthwise=True, bn_relu="relu", kernel=3, stride=1, dilation=1))
	Filters.append(FilterParams(
					tvm.placeholder((1, 1, 128, 128), name='Conv2dFilter_0'),
					depthwise=False, bn_relu="relu", kernel=1, stride=1, dilation=1))

	stages, data = fused_convs(Input, Filters)
	print(stages)
	print(data)