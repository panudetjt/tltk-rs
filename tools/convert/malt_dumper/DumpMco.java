import java.io.*;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.*;
import org.maltparser.core.symbol.hash.HashSymbolTable;
import org.maltparser.core.symbol.hash.HashSymbolTableHandler;
import org.maltparser.ml.lib.FeatureMap;
import org.maltparser.ml.lib.MaltLiblinearModel;

public class DumpMco {
    static void escapeJson(String s, StringBuilder sb) {
        if (s == null) {
            sb.append("null");
            return;
        }
        sb.append('"');
        int len = s.length();
        for (int i = 0; i < len; i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\b': sb.append("\\b"); break;
                case '\f': sb.append("\\f"); break;
                case '\n': sb.append("\\n"); break;
                case '\r': sb.append("\\r"); break;
                case '\t': sb.append("\\t"); break;
                default:
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
            }
        }
        sb.append('"');
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("Usage: DumpMco <mco-path> [mco-sha256]");
            System.exit(1);
        }
        String mcoPath = args[0];
        String sha256 = args.length > 1 ? args[1] : "";

        File mcoFile = new File(mcoPath);
        String modelName = mcoFile.getName();
        if (modelName.endsWith(".mco")) {
            modelName = modelName.substring(0, modelName.length() - 4);
        }

        String conllxXml = null;
        String featureSpecXml = null;
        Map<String, String> options = new TreeMap<>();
        byte[] symBytes = null;
        FeatureMap featureMap = null;
        MaltLiblinearModel liblinearModel = null;
        String decisionModelName = "odm0";

        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(mcoFile))) {
            ZipEntry e;
            while ((e = zis.getNextEntry()) != null) {
                String name = e.getName();
                if (name.endsWith("conllx.xml")) {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    zis.transferTo(baos);
                    conllxXml = new String(baos.toByteArray(), StandardCharsets.UTF_8);
                } else if (name.endsWith("NivreEager.xml")) {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    zis.transferTo(baos);
                    featureSpecXml = new String(baos.toByteArray(), StandardCharsets.UTF_8);
                } else if (name.endsWith("savedoptions.sop")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis, StandardCharsets.UTF_8));
                    String line;
                    while ((line = br.readLine()) != null) {
                        line = line.trim();
                        if (line.isEmpty()) continue;
                        String[] parts = line.split("\t", -1);
                        if (parts.length >= 3) {
                            String task = parts[1];
                            String param = parts[2];
                            String val = parts.length > 3 ? parts[3] : "";
                            options.put(task + "_" + param, val);
                        }
                    }
                } else if (name.endsWith(".sym")) {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    zis.transferTo(baos);
                    symBytes = baos.toByteArray();
                } else if (name.endsWith(".map")) {
                    int slash = name.lastIndexOf('/');
                    String base = slash >= 0 ? name.substring(slash + 1) : name;
                    int dot = base.indexOf('.');
                    if (dot > 0) decisionModelName = base.substring(0, dot);

                    ObjectInputStream ois = new ObjectInputStream(zis);
                    featureMap = (FeatureMap) ois.readObject();
                } else if (name.endsWith(".moo")) {
                    ObjectInputStream ois = new ObjectInputStream(zis);
                    liblinearModel = (MaltLiblinearModel) ois.readObject();
                }
            }
        }

        // Parse feature strings from featureSpecXml (<feature>...</feature>)
        List<String> featureStrings = new ArrayList<>();
        if (featureSpecXml != null) {
            int idx = 0;
            while (true) {
                int start = featureSpecXml.indexOf("<feature>", idx);
                if (start == -1) break;
                int end = featureSpecXml.indexOf("</feature>", start);
                if (end == -1) break;
                featureStrings.add(featureSpecXml.substring(start + 9, end).trim());
                idx = end + 10;
            }
        }

        // Load symbol tables with HashSymbolTableHandler
        HashSymbolTableHandler symHandler = new HashSymbolTableHandler();
        if (symBytes != null) {
            try (InputStreamReader isr = new InputStreamReader(new ByteArrayInputStream(symBytes), StandardCharsets.UTF_8)) {
                symHandler.load(isr);
            }
        }

        // Build output JSON
        StringBuilder sb = new StringBuilder(10 * 1024 * 1024);
        sb.append("{\n");
        sb.append("  \"format\": \"tltk-malt\",\n");
        sb.append("  \"format_version\": 1,\n");
        sb.append("  \"model_name\": "); escapeJson(modelName, sb); sb.append(",\n");
        sb.append("  \"maltparser_version\": \"1.9.2\",\n");
        sb.append("  \"source_mco_sha256\": "); escapeJson(sha256, sb); sb.append(",\n");

        // options
        sb.append("  \"options\": {\n");
        int optCount = 0;
        int optTotal = options.size();
        for (Map.Entry<String, String> entry : options.entrySet()) {
            optCount++;
            sb.append("    "); escapeJson(entry.getKey(), sb); sb.append(": "); escapeJson(entry.getValue(), sb);
            if (optCount < optTotal) sb.append(",");
            sb.append("\n");
        }
        sb.append("  },\n");

        // data_format_xml
        sb.append("  \"data_format_xml\": "); escapeJson(conllxXml, sb); sb.append(",\n");

        // feature_spec_xml
        sb.append("  \"feature_spec_xml\": "); escapeJson(featureSpecXml, sb); sb.append(",\n");

        // feature_strings
        sb.append("  \"feature_strings\": [\n");
        for (int i = 0; i < featureStrings.size(); i++) {
            sb.append("    "); escapeJson(featureStrings.get(i), sb);
            if (i + 1 < featureStrings.size()) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ],\n");

        // symbol_tables
        sb.append("  \"symbol_tables\": [\n");
        List<String> tableNames = new ArrayList<>(symHandler.getSymbolTableNames());
        Collections.sort(tableNames);
        Field fCodeMap = HashSymbolTable.class.getDeclaredField("codeSymbolMap");
        fCodeMap.setAccessible(true);

        for (int tIdx = 0; tIdx < tableNames.size(); tIdx++) {
            String tName = tableNames.get(tIdx);
            HashSymbolTable st = (HashSymbolTable) symHandler.getSymbolTable(tName);
            Map<Integer, String> codeMap = (Map<Integer, String>) fCodeMap.get(st);
            List<Integer> codes = new ArrayList<>(codeMap.keySet());
            Collections.sort(codes);

            sb.append("    {\n");
            sb.append("      \"name\": "); escapeJson(tName, sb); sb.append(",\n");
            sb.append("      \"header\": [");
            escapeJson(Integer.toString(st.getCategory()), sb); sb.append(", ");
            escapeJson(Integer.toString(st.getType()), sb); sb.append(", ");
            escapeJson(st.getNullValueStrategy() != null ? st.getNullValueStrategy() : "", sb);
            sb.append("],\n");

            sb.append("      \"symbols\": [");
            for (int sIdx = 0; sIdx < codes.size(); sIdx++) {
                int code = codes.get(sIdx);
                String sym = codeMap.get(code);
                if (sIdx > 0) sb.append(", ");
                sb.append("{\"code\": ").append(code).append(", \"symbol\": ");
                escapeJson(sym, sb);
                sb.append("}");
            }
            sb.append("]\n");
            sb.append("    }");
            if (tIdx + 1 < tableNames.size()) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ],\n");

        // decision_models
        sb.append("  \"decision_models\": [\n");
        sb.append("    {\n");
        sb.append("      \"name\": "); escapeJson(decisionModelName, sb); sb.append(",\n");

        // feature_map
        Field fCounter = FeatureMap.class.getDeclaredField("featureCounter");
        fCounter.setAccessible(true);
        int nextIndex = (Integer) fCounter.get(featureMap);

        Field fMap = FeatureMap.class.getDeclaredField("map");
        fMap.setAccessible(true);
        Map<Long, Integer> rawMap = (Map<Long, Integer>) fMap.get(featureMap);
        List<Map.Entry<Long, Integer>> sortedEntries = new ArrayList<>(rawMap.entrySet());
        sortedEntries.sort(Comparator.comparingInt(Map.Entry::getValue));

        sb.append("      \"feature_map\": {\n");
        sb.append("        \"next_index\": ").append(nextIndex).append(",\n");
        sb.append("        \"entries\": [\n");
        for (int eIdx = 0; eIdx < sortedEntries.size(); eIdx++) {
            Map.Entry<Long, Integer> entry = sortedEntries.get(eIdx);
            sb.append("          [\"").append(entry.getKey()).append("\", ").append(entry.getValue()).append("]");
            if (eIdx + 1 < sortedEntries.size()) sb.append(",");
            sb.append("\n");
        }
        sb.append("        ]\n");
        sb.append("      },\n");

        // liblinear
        Field fw = MaltLiblinearModel.class.getDeclaredField("w");
        fw.setAccessible(true);
        double[][] w = (double[][]) fw.get(liblinearModel);

        int nrClass = liblinearModel.getNrClass();
        int nrFeature = liblinearModel.getNrFeature();
        double bias = liblinearModel.getBias();
        Field fSolver = MaltLiblinearModel.class.getDeclaredField("solverType");
        fSolver.setAccessible(true);
        String solver = fSolver.get(liblinearModel).toString();
        int[] labels = liblinearModel.getLabels();

        sb.append("      \"liblinear\": {\n");
        sb.append("        \"solver\": "); escapeJson(solver, sb); sb.append(",\n");
        sb.append("        \"nr_class\": ").append(nrClass).append(",\n");
        sb.append("        \"labels\": [");
        for (int lIdx = 0; lIdx < labels.length; lIdx++) {
            if (lIdx > 0) sb.append(", ");
            sb.append(labels[lIdx]);
        }
        sb.append("],\n");
        sb.append("        \"nr_feature\": ").append(nrFeature).append(",\n");
        sb.append("        \"bias\": ").append(Double.toString(bias)).append(",\n");

        // w: sparse per class
        sb.append("        \"w\": [\n");
        for (int c = 0; c < nrClass; c++) {
            sb.append("          [");
            boolean first = true;
            for (int f = 0; f < nrFeature; f++) {
                if (w[f] != null && c < w[f].length && w[f][c] != 0.0) {
                    if (!first) sb.append(", ");
                    first = false;
                    sb.append("[").append(f).append(", ").append(Double.toString(w[f][c])).append("]");
                }
            }
            sb.append("]");
            if (c + 1 < nrClass) sb.append(",");
            sb.append("\n");
        }
        sb.append("        ]\n");
        sb.append("      }\n");
        sb.append("    }\n");
        sb.append("  ]\n");
        sb.append("}\n");

        byte[] utf8Bytes = sb.toString().getBytes(StandardCharsets.UTF_8);
        System.out.write(utf8Bytes);
        System.out.flush();
    }
}
