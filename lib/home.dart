import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;
import 'package:image/image.dart' as imglib;
import 'package:flutter/painting.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  List? _result;
  bool _imageSelected = false;
  bool _loading = false;
  final _imagePicker = ImagePicker();
  //---------------------------
  late tfl.Interpreter _interpreter;
  late List inputShape;
  late List outputShape;
  late tfl.TensorType inputType;
  late tfl.TensorType outputType;

  //-image specs---------------------------
  double x = 0;
  late double y = 0;
  late double h = 0;
  late double w = 0;
  late double cls = 0;
  late double conf = 0;
  //------------------------------------

  @override
  Future getImage(ImageSource source) async {
    final image = await ImagePicker().pickImage(source: source);
    if (image == null) {
      return;
    }
    final imageTemporary = File(image.path);
    setState(() {
      _image = imageTemporary;
      _imageSelected = false;
      _result = null;
    });
    classifyImage(_image);
  }

  Future classifyImage(File? image) async {
    if (image == null) {
      return;
    }
    final imageBytes = await image.readAsBytes();
    var inputTensor = preProcessImage(imageBytes);
    var outputTensor = List.filled(1 * 10647 * 6, 0.0).reshape([1, 10647, 6]);

    _interpreter.run(inputTensor, outputTensor);
    List<double> detections = postProcess(outputTensor); // ---(1)
    print("------output detection best-----------$detections");

    setState(() {
      conf = detections[4];
      _loading = false;
      _result = detections;
    });
  }

  List<double> postProcess(List<dynamic> outputTensor) {
    double maxConfidence = 0.0;
    List<double>? maxConfidenceDetection;
    for (int i = 0; i < outputTensor[0].length; i++) {
      List<double> prediction = outputTensor[0][i];
      double x = prediction[0];
      double y = prediction[1];
      double w = prediction[2];
      double h = prediction[3];
      double conf = prediction[4];

      if (conf > maxConfidence) {
        maxConfidence = conf;
        maxConfidenceDetection = [x, y, w, h, conf, prediction[5]];
      }
    }
    return maxConfidenceDetection ?? [];
  }

  List<List<List<List<double>>>> preProcessImage(Uint8List imageBytes) {
    imglib.Image img = imglib.decodeImage(imageBytes)!;
    imglib.Image resizedImage = imglib.copyResize(img, width: 416, height: 416);

    List<List<List<List<double>>>> inputValues = List.generate(1, (batchIndex) {
      List<List<List<double>>> batch = [];
      for (int row = 0; row < 416; row++) {
        List<List<double>> rowValues = [];
        for (int col = 0; col < 416; col++) {
          List<double> pixelValues = [];

          int pixel = resizedImage.getPixel(col, row);
          double r = imglib.getRed(pixel) / 255.0;
          double g = imglib.getGreen(pixel) / 255.0;
          double b = imglib.getBlue(pixel) / 255.0;

          pixelValues.add(r);
          pixelValues.add(g);
          pixelValues.add(b);

          rowValues.add(pixelValues);
        }
        batch.add(rowValues);
      }
      return batch;
    });

    return inputValues;
  }

  // Input shape: [1, 416, 416, 3]
  // Output shape: [1, 10647, 6]
  Future<void> loadModel() async {
    _interpreter = await tfl.Interpreter.fromAsset("assets/best-fp16.tflite");
    inputShape = _interpreter.getInputTensor(0).shape;
    outputShape = _interpreter.getOutputTensor(0).shape;
    print('--------------------------Input shape: $inputShape');
    print('--------------------------Output shape: $outputShape');
    inputType = _interpreter.getInputTensor(0).type;
    outputType = _interpreter.getOutputTensor(0).type;
    print('--------------------------Input type: $inputType');
    print('--------------------------Output type: $outputType');
  }

  void initState() {
    super.initState();
    _loading = true;
    loadModel().then((value) {
      setState(() {
        _loading = false;
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Tomato Detector'),
      ),
      body: Center(
        child: Column(
          children: [
            _image != null
                ? Stack(
                    children: [
                      Image.file(
                        _image!,
                        width: 416,
                        height: 416,
                        fit: BoxFit.cover,
                      ),
                      if (_result != null)
                        Positioned.fill(
                          child: CustomPaint(
                            painter: BoundingBoxPainter(
                              imageSize: const Size(416, 416),
                              detection: _result!,
                            ),
                          ),
                        ),
                    ],
                  )
                : Container(),
            CustomButton(
                'Pick from Gallery', () => getImage(ImageSource.gallery)),
            CustomButton('Open Camera', () => getImage(ImageSource.camera)),
            if (_result != null)
              Text(
                conf >= 0.5 ? 'Confidence: ${conf * 100}' : 'No Detections',
                style: const TextStyle(fontSize: 20),
              ),
          ],
        ),
      ),
    );
  }
}

class CustomButton extends StatelessWidget {
  final String title;
  final VoidCallback onClick;

  CustomButton(this.title, this.onClick);

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 280,
      child: ElevatedButton(
        onPressed: onClick,
        child: Align(
          alignment: Alignment.center,
          child: Text(title),
        ),
      ),
    );
  }
}

//--------------------bounding boxes------------------------
void drawBoundingBox(Canvas canvas, Size imageSize, List detection) {
  double x = detection[0];
  double y = detection[1];
  double w = detection[2];
  double h = detection[3];
  if (detection[4] >= 0.5) {
    // Scale the coordinates to match the image dimensions
    double imageWidth = imageSize.width;
    double imageHeight = imageSize.height;

    x *= imageWidth;
    y *= imageHeight;
    w *= imageWidth;
    h *= imageHeight;

    double left = x - w / 2;
    double top = y - h / 2;
    double right = x + w / 2;
    double bottom = y + h / 2;

    // Create a paint object to define the bounding box style
    Paint paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), paint);
  } else {
    print("No detections");
  }
}

class BoundingBoxPainter extends CustomPainter {
  final Size imageSize;
  final List detection;

  BoundingBoxPainter({
    required this.imageSize,
    required this.detection,
  });

  @override
  void paint(Canvas canvas, Size size) {
    drawBoundingBox(canvas, imageSize, detection);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return false;
  }
}
