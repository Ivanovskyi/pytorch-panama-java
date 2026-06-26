package org.example;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.List;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;

public class Main {

    static final int EMBED_DIM = 384;

    public static void main(String[] args) throws Throwable {

        // ----------------------------
        // 1. Load native library
        // ----------------------------
        System.load("/home/admin/torch_demo/build/libsearch_engine.so");

        Linker linker = Linker.nativeLinker();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        try (Arena arena = Arena.ofConfined()) {

            // ----------------------------
            // 2. Load model into C++
            // ----------------------------
            MethodHandle loadModel = linker.downcallHandle(
                    lookup.find("load_model").orElseThrow(),
                    FunctionDescriptor.of(ValueLayout.JAVA_BOOLEAN, ValueLayout.ADDRESS)
            );

            String modelPath = "/home/admin/torch_demo/model.pt";
            MemorySegment modelSeg = arena.allocateFrom(modelPath);

            boolean ok = (boolean) loadModel.invokeExact((MemorySegment) modelSeg);

            if (!ok) {
                System.out.println("❌ Model failed to load");
                return;
            }

            System.out.println("✅ Model loaded successfully");

            // ----------------------------
            // 3. get_embedding function
            // ----------------------------
            MethodHandle getEmbedding = linker.downcallHandle(
                    lookup.find("get_embedding").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            ValueLayout.ADDRESS,   // input_ids
                            ValueLayout.ADDRESS,   // attention_mask
                            ValueLayout.JAVA_LONG, // length
                            ValueLayout.ADDRESS    // output vector
                    )
            );

            // ----------------------------
            // 4. Load tokenizer
            // ----------------------------
            HuggingFaceTokenizer tokenizer =
                    HuggingFaceTokenizer.newInstance(
                            "sentence-transformers/all-MiniLM-L6-v2"
                    );

            // ----------------------------
            // 5. Demo dataset
            // ----------------------------
            List<String> products = List.of(
                    "Mineral sparkling water Coca-Cola 0.5l",
                    "Smartphone Apple iPhone 15 Pro Max 256GB",
                    "Winter men's hooded jacket down jacket",
                    "Fresh croissant with chocolate filling"
            );

            float[][] productVectors = new float[products.size()][EMBED_DIM];

            for (int i = 0; i < products.size(); i++) {
                productVectors[i] = embed(
                        products.get(i),
                        tokenizer,
                        arena,
                        getEmbedding
                );
            }

            // ----------------------------
            // 6. Query
            // ----------------------------
            String query = "want to drink";
            System.out.println("\n🔎 Query: " + query);

            float[] queryVec = embed(query, tokenizer, arena, getEmbedding);

            // ----------------------------
            // 7. Similarity search
            // ----------------------------
            System.out.println("\n📊 Results:");

            for (int i = 0; i < products.size(); i++) {
                double sim = cosine(queryVec, productVectors[i]);

                System.out.printf("- %s → %.2f%%\n",
                        products.get(i),
                        sim * 100
                );
            }
        }
    }

    // ----------------------------
    // EMBEDDING FUNCTION
    // ----------------------------
    static float[] embed(
            String text,
            HuggingFaceTokenizer tokenizer,
            Arena arena,
            MethodHandle getEmbedding
    ) throws Throwable {

        Encoding enc = tokenizer.encode(text);

        long[] ids = enc.getIds();
        long[] mask = enc.getAttentionMask();
        long len = ids.length;

        MemorySegment idsSeg =
                arena.allocateFrom(ValueLayout.JAVA_LONG, ids);

        MemorySegment maskSeg =
                arena.allocateFrom(ValueLayout.JAVA_LONG, mask);

        MemorySegment outSeg =
                arena.allocate(ValueLayout.JAVA_FLOAT, EMBED_DIM);

        getEmbedding.invokeExact(
                (MemorySegment) idsSeg,
                (MemorySegment) maskSeg,
                len,
                (MemorySegment) outSeg
        );

        return outSeg.toArray(ValueLayout.JAVA_FLOAT);
    }

    // ----------------------------
    // COSINE SIMILARITY
    // ----------------------------
    static double cosine(float[] a, float[] b) {

        double dot = 0;
        double na = 0;
        double nb = 0;

        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }

        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }
}