	C??feY@C??feY@!C??feY@      ??!       "n
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
	??I?G@??I?G@!??I?G@      ??!       "	Y??WW@Y??WW@!Y??WW@*      ??!       2	?L?J????L?J???!?L?J???:	z9??c? @z9??c? @!z9??c? @B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?(?H* @y??Z???V@