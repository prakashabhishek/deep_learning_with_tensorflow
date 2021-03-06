?	C??feY@C??feY@!C??feY@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-C??feY@??I?G@1Y??WW@A?L?J???Iz9??c? @*	?Q???S@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???/fK??!???iR?K@)???/fK??1???iR?K@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism҉Sͬ??!?g/w?U@)?R?h??1G?? @@:Preprocessing2F
Iterator::Model?ԕ??<??!      Y@)?X??+???1S?t?FT)@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?(?H* @Q??Z???V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??I?G@??I?G@!??I?G@      ??!       "	Y??WW@Y??WW@!Y??WW@*      ??!       2	?L?J????L?J???!?L?J???:	z9??c? @z9??c? @!z9??c? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?(?H* @y??Z???V@?"-
IteratorGetNext/_2_Recv?F?֮9??!?F?֮9??"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block3/unit_1/bottleneck_v2/shortcut/Conv2DConv2D??esGG??!?ĕ????0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block4/unit_1/bottleneck_v2/shortcut/Conv2DConv2D??`.??!n??vW??0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block1/unit_1/bottleneck_v2/addAddV22GPJ?ݐ?!???<G??"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block1/unit_2/bottleneck_v2/addAddV2QqUcؐ?!G?+@b??"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block3/unit_4/bottleneck_v2/conv3/Conv2DConv2D?*Ȯ??!I?x??0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block3/unit_3/bottleneck_v2/conv3/Conv2DConv2D?F??d???!&???E???0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block3/unit_5/bottleneck_v2/conv3/Conv2DConv2D_\=?????!?2?0????0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block3/unit_2/bottleneck_v2/conv3/Conv2DConv2D?^???!??????0"?
?sequential/feature_extractor_layer/StatefulPartitionedCall/StatefulPartitionedCall/StatefulPartitionedCall/predict/resnet_v2_50/block2/unit_1/bottleneck_v2/shortcut/Conv2DConv2D?Px	?f??!??Fr????0Q      Y@Y????Ĭ@a??U?3?W@q8)?Qw?M@yȘ޲6?p?"?

both?Your program is POTENTIALLY input-bound because 6.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?59.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 