	??F???@??F???@!??F???@		???"@	???"@!	???"@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??F???@?zM
J??AD?????@Y???!??rEagerKernelExecute 0*	??Mb`y@2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??x??[??!?t??e8@)??x??[??1?t??e8@:Preprocessing2F
Iterator::Modell??????!???+C@)vP??W??1??y?k7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatE??S???!???8?D@)X??V?ı?1???n?1@:Preprocessing2U
Iterator::Model::ParallelMapV2?\R????!?A?-@)?\R????1?A?-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?D?A???!z??Er?N@)?W}w??1M??!??"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?c???Ȥ?!????q?#@)0H?????1?h!??b@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice.9????!?*???8@).9????1?*???8@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap̸???s??!??-?%@)iSu?l?j?1??ZⱫ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s4.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9	???"@I??,??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?zM
J???zM
J??!?zM
J??      ??!       "      ??!       *      ??!       2	D?????@D?????@!D?????@:      ??!       B      ??!       J	???!?????!??!???!??R      ??!       Z	???!?????!??!???!??b      ??!       JCPU_ONLYY	???"@b q??,??V@