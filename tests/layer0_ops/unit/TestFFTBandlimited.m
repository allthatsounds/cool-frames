classdef TestFFTBandlimited < matlab.unittest.TestCase
    % TestFFTBandlimited: unit tests for comp_filterbank_fftbl
    %
    % Correct calling convention (from comp_filterbank.m):
    %   c = comp_filterbank_fftbl(F, G, foff, a, realonly)
    %       F       : fft(f), size [L x W]
    %       G       : cell{M} of SHORT DFT responses (numel(G{m}) < L)
    %       foff    : [M x 1] frequency offsets (starting bin for each G{m})
    %       a       : subsampling factors [M x 1] or [M x 2] for fractional
    %       realonly: [M x 1] logical -- 1 = add conjugate mirror band
    %
    % Uses synthetic rectangular band-limited filters to avoid any
    % dependence on audfilters / comp_filterbank_pre path setup.

    properties
        Ls          % signal length
        M           % number of bands
        bw          % bandwidth of each band (in DFT bins)
        G_bl        % cell{M} of band-limited DFT responses (length bw each)
        foff_bl     % [M x 1] frequency offsets (starting bin per band)
        realonly_bl % [M x 1] realonly flags (all 0 = complex-valued)
        a_bl        % [M x 1] subsampling factors
        noise_real
        noise_complex
        zeros_sig
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 1024;
            testCase.M  = 4;
            testCase.bw = 8;   % bandwidth of each band in DFT bins

            % Build M synthetic rectangular band-limited filters
            M     = testCase.M;
            bw    = testCase.bw;
            Ls    = testCase.Ls;
            step  = floor(Ls / (2 * M));
            G_tmp  = cell(M, 1);
            foff_v = zeros(M, 1);
            ro_v   = zeros(M, 1);   % not realonly
            a_v    = ones(M, 1);    % no subsampling
            for m = 1 : M
                G_tmp{m}  = ones(bw, 1);           % rectangular BL filter
                foff_v(m) = (m - 1) * step;        % evenly spaced frequency offsets
            end
            testCase.G_bl        = G_tmp;
            testCase.foff_bl     = foff_v;
            testCase.realonly_bl = ro_v;
            testCase.a_bl        = a_v;

            testCase.noise_real    = randn(testCase.Ls, 1);
            testCase.noise_complex = randn(testCase.Ls, 1) + 1i*randn(testCase.Ls, 1);
            testCase.zeros_sig     = zeros(testCase.Ls, 1);
        end
    end

    methods(Test)

        function testFFTBLZeroInput(testCase)
            % Zero signal -> all subbands zero
            F = fft(testCase.zeros_sig);
            c = comp_filterbank_fftbl(F, testCase.G_bl, testCase.foff_bl, ...
                                      testCase.a_bl, testCase.realonly_bl);
            for m = 1:length(c)
                testCase.verifyEqual(c{m}, zeros(size(c{m})), 'AbsTol', 1e-14, ...
                    sprintf('Subband %d not zero for zero input', m));
            end
        end

        function testFFTBLOutputSizes(testCase)
            % Each subband should have N = Ls/a rows and at least 1 column
            F = fft(testCase.noise_real);
            c = comp_filterbank_fftbl(F, testCase.G_bl, testCase.foff_bl, ...
                                      testCase.a_bl, testCase.realonly_bl);
            testCase.verifyGreaterThan(length(c), 0, 'No subbands returned');
            for m = 1:length(c)
                testCase.verifyGreaterThanOrEqual(size(c{m}, 1), 1, ...
                    sprintf('Subband %d has zero rows', m));
            end
        end

        function testFFTBLRealSignal(testCase)
            % Real input: M subbands returned without error
            F = fft(testCase.noise_real);
            c = comp_filterbank_fftbl(F, testCase.G_bl, testCase.foff_bl, ...
                                      testCase.a_bl, testCase.realonly_bl);
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testFFTBLComplexSignal(testCase)
            % Complex input: M subbands returned without error
            F = fft(testCase.noise_complex);
            c = comp_filterbank_fftbl(F, testCase.G_bl, testCase.foff_bl, ...
                                      testCase.a_bl, testCase.realonly_bl);
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testFFTBLOutputLength(testCase)
            % Number of returned subbands equals M (number of band-limited filters)
            F = fft(testCase.noise_real);
            c = comp_filterbank_fftbl(F, testCase.G_bl, testCase.foff_bl, ...
                                      testCase.a_bl, testCase.realonly_bl);
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testFFTBLRealOnly(testCase)
            % Setting realonly=1 is accepted without error for one band
            G_ro   = testCase.G_bl(1);
            foff_ro = testCase.foff_bl(1);
            a_ro   = testCase.a_bl(1);
            ro_ro  = 1;   % realonly = true for this single band
            F = fft(testCase.noise_real);
            c = comp_filterbank_fftbl(F, G_ro, foff_ro, a_ro, ro_ro);
            testCase.verifyEqual(length(c), 1);
        end

    end

end
