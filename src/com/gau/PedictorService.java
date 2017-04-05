package com.gau;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

import javax.ws.rs.Consumes;
import javax.ws.rs.GET;
import javax.ws.rs.POST;
import javax.ws.rs.Path;
import javax.ws.rs.Produces;
import javax.ws.rs.core.MediaType;
import javax.ws.rs.core.Response;

import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;


//http://localhost:9090/DefectPredictorApi/api/predictor_service
@Path("/")

public class PedictorService {
	@POST
	@Path("/predictor_service")
	@Consumes(MediaType.APPLICATION_JSON)
	public Response predictionAnalysis(InputStream incomingData) throws ParseException {
		StringBuilder sBuilder = new StringBuilder();
		try {
			BufferedReader in = new BufferedReader(new InputStreamReader(incomingData));
			String line = null;
			while ((line = in.readLine()) != null) {
				sBuilder.append(line);
			}
		} catch (Exception e) {
			System.out.println("Error Parsing: - ");
		}
		
		if(sBuilder.length()<0 || sBuilder.toString().isEmpty())
		return Response.status(204).entity("An empty Json String recevied").build();
		
		// return HTTP response 200 in case of success
		return Response.status(200).entity(analyseMatrics(sBuilder.toString())).build();
	}

	@GET
	@Path("/verify")
	@Produces(MediaType.TEXT_PLAIN)
	public Response verifyRESTService(InputStream incomingData) {
		String result = "PredictorRESTService Successfully started..";
		// return HTTP response 200 in case of success
		return Response.status(200).entity(result).build();
	}

	public String analyseMatrics(String initialReqJsonString) throws ParseException {

		SparkSession sparkSession = SparkSession.builder().appName("DP-App").master("local[2]").getOrCreate();
		System.out.println("Json Received : "+ initialReqJsonString);
		
		JsonParser jsonParser = new JsonParser();
		JsonArray initialReqJson = (JsonArray) jsonParser.parse(initialReqJsonString);
		
		initialReqJsonString = getMatricsValuesFromJson(initialReqJson);
		System.out.println("Json Refined For SourceMeter Process: "+initialReqJsonString);
		
		String [] marticsKeySets = getMatricsKeySet(initialReqJsonString);
		
		List<String> data = Arrays.asList(initialReqJsonString);
		Dataset<Row> dataframeTemp = sparkSession.createDataset(data, Encoders.STRING()).toDF().withColumnRenamed("_1","value");
		Dataset<String> df1 = dataframeTemp.as(Encoders.STRING());
		Dataset<Row> dataFrame = sparkSession.read().json(df1.javaRDD());
		LogisticRegressionModel model = null;
		if(new File("c:/temp/spark-models/lr-model.dat").exists())
		model = LogisticRegressionModel.load("c:/temp/spark-models/lr-model.dat");
		else return "[{\"error\":\"spark-models/lr-model.dat file not found\"}]";
		dataFrame.printSchema();
		//VectorAssembler va = new VectorAssembler().setInputCols(new String[] { "McCC", "CLOC", "PDA", "PUA", "LOC", "LLOC" }).setOutputCol("features");
		VectorAssembler va = new VectorAssembler().setInputCols(marticsKeySets).setOutputCol("features");
		try{
		dataFrame = va.transform(dataFrame);
		}catch(Exception exp){ return "[{\"error\":\"Defect Analysis process unsuccessful the data may be not correct.\"}]";}
		dataFrame = model.transform(dataFrame);
		dataFrame = dataFrame.drop("features").drop("rawPrediction");
		dataFrame.show();
		String resultJson = convertDatasetToJson(dataFrame, initialReqJson).toString();
		System.out.println("Json Result from the defect Analysis: "+resultJson);
		if(!resultJson.isEmpty())
		return resultJson;
		else return "[{\"error\":\"An empty dataset received\"}]";
			
	}

	
	private String[] getMatricsKeySet(String initialReqJsonString) throws ParseException {
		
		JSONParser parser = new JSONParser();
		JSONArray jsonArray = (JSONArray) parser.parse(initialReqJsonString);
		JSONObject jsonObject = (JSONObject) jsonArray.get(1);
		
		Object[] object = jsonObject.keySet().toArray();
		String[] stringArray = Arrays.copyOf(object, object.length, String[].class);
		
		return stringArray;
	}

	private JsonArray convertDatasetToJson(Dataset dataFrame , JsonArray initialReqJArray ) {
		List<Row> list = dataFrame.collectAsList();
		String json = new Gson().toJson(list);
		System.out.println(json);
		JsonParser parser = new JsonParser();
		JsonArray analysisJsonArray = (JsonArray) parser.parse(json);
		JsonArray resultJsonArray = new JsonArray();

		JsonObject schemaJsonObject = (JsonObject) ((JsonObject) analysisJsonArray.get(0)).get("schema");
		JsonArray fieldsJsonArray = schemaJsonObject.getAsJsonArray("fields");
		String[] headersArray = new String[fieldsJsonArray.size()];
		
		for (int i = 0; i < analysisJsonArray.size(); i++) {
			JsonArray valuesJsonArray = (JsonArray) ((JsonObject) analysisJsonArray.get(i)).get("values");
			JsonObject tempObj = new JsonObject();
			tempObj.add("ID", initialReqJArray.get(i).getAsJsonObject().get("ID"));
			tempObj.add("probability", valuesJsonArray.get(valuesJsonArray.size()-2));
			tempObj.add("prediction", valuesJsonArray.get(valuesJsonArray.size()-1));
			resultJsonArray.add(tempObj);
		}
		
		
		return resultJsonArray;
	}

	
	public String getMatricsValuesFromJson(JsonArray jsonMainArray)
	{
		
		
		JsonArray mainResultJArray = new JsonArray();
		for(int i = 0; i<jsonMainArray.size(); i++)
		{
			JsonObject temp = (JsonObject) jsonMainArray.get(i);
			JsonObject matrix = (JsonObject) temp.get("matrics");
			mainResultJArray.add(matrix);
		}
		return mainResultJArray.toString();
	}
	
}
