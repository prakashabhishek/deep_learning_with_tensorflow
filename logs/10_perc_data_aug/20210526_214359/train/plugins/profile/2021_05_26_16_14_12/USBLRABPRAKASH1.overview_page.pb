?	Ωd ?R?@Ωd ?R?@!Ωd ?R?@	Dq9?S???Dq9?S???!Dq9?S???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Ωd ?R?@?q6??A???*!?@Y???v8@rEagerKernelExecute 0*	??阮?@2f
/Iterator::Model::MaxIntraOpParallelism::BatchV26?;N?e8@!? ?2k?X@)8H???,@1ƃ??$+M@:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::ParallelMapV2 ?3?/.?"@!E?OH??B@)?3?/.?"@1E?OH??B@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip ?Q+L??#@!l?֛
D@)??w??1?n??????:Preprocessing2?
\Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice 3??bb???!?ZX3:??)3??bb???1?ZX3:??:Preprocessing2?
OIterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::ParallelMapV2 ???'???!??s
????)???'???1??s
????:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle 3R臭$@!?}߱MD@)\ qW???1/?(ѐ???:Preprocessing2?
\Iterator::Model::MaxIntraOpParallelism::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice 4??O??!M*??P??)4??O??1M*??P??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism????i8@!??sD??X@)N'??rJ??1?(+F???:Preprocessing2F
Iterator::Model??ฌk8@!!?;?:?X@)??3?ތz?1?W?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Dq9?S???I;???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?q6???q6??!?q6??      ??!       "      ??!       *      ??!       2	???*!?@???*!?@!???*!?@:      ??!       B      ??!       J	???v8@???v8@!???v8@R      ??!       Z	???v8@???v8@!???v8@b      ??!       JCPU_ONLYYDq9?S???b q;???X@Y      Y@q?xiD???"?
device?Your program is NOT input-bound because only 1.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 