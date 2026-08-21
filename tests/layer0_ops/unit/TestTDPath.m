classdef TestTDPath < matlab.unittest.TestCase
    % TestTDPath: unit tests for comp_filterbank_td and comp_ifilterbank_td
    %
    % Correct calling conventions (from source):
    %   c = comp_filterbank_td(f, g, a, offset, ext)
    %       f      : time-domain signal [L x W]
    %       g      : cell{M} of impulse response vectors
    %       a      : subsampling factors [M x 1]
    %       offset : filter offset vector [M x 1]; offset=0 => causal (skip=0)
    %       ext    : boundary extension string ('per', 'zpd', ...)
    %
    %   f = comp_ifilterbank_td(c, g, a, Ls, offset, ext)
    %       c      : cell{M} of coefficient matrices
    %       g      : same cell of impulse response vectors
    %       a      : upsampling factors [M x 1]
    %       Ls     : desired output length
    %       offset : filter offset vector [M x 1]
    %       ext    : boundary extension string
    %
    % Offset convention: offset = 0 means skip = -offset = 0, i.e. causal filter
    % with the "reference" sample at position 0 in the filter support.

    properties
        Ls
        M           % number of filters
        Lh          % filter length
        a           % subsampling factors [M x 1]
        G_td        % cell{M} of FIR impulse responses
        offset      % filter offsets [M x 1] (all zero = causal)
        noise_real
        zeros_sig
        impulse
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 256;
            testCase.M  = 3;
            testCase.Lh = 16;          % FIR filter length
            testCase.a  = [2; 4; 4];   % subsampling factors

            % Build M short causal FIR filters (Hann window)
            M     = testCase.M;
            Lh    = testCase.Lh;
            G_tmp = cell(M, 1);
            for m = 1 : M
                G_tmp{m} = firwin('hann', Lh);
            end
            testCase.G_td   = G_tmp;
            testCase.offset = zeros(testCase.M, 1);   % causal (offset = 0)

            testCase.noise_real = randn(testCase.Ls, 1);
            testCase.zeros_sig  = zeros(testCase.Ls, 1);
            testCase.impulse    = zeros(testCase.Ls, 1);
            testCase.impulse(1) = 1;
        end
    end

    methods(Test)

        function testTDZeroInput(testCase)
            % Zero signal -> all subband outputs zero
            c = comp_filterbank_td(testCase.zeros_sig, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');
            for m = 1:testCase.M
                testCase.verifyEqual(c{m}, zeros(size(c{m})), 'AbsTol', 1e-14, ...
                    sprintf('Subband %d not zero for zero input', m));
            end
        end

        function testTDImpulseResponse(testCase)
            % Impulse input -> at least some subbands are non-zero
            c = comp_filterbank_td(testCase.impulse, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');
            non_zero = sum(cellfun(@(x) any(x(:) ~= 0), c));
            testCase.verifyGreaterThan(non_zero, 0, 'No subband responded to impulse');
        end

        function testTDOutputSizes(testCase)
            % Each c{m} has ceil(Ls/a(m)) rows for 'per' extension
            c = comp_filterbank_td(testCase.noise_real, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');
            for m = 1:testCase.M
                expected = ceil(testCase.Ls / testCase.a(m));
                testCase.verifyEqual(size(c{m}, 1), expected, ...
                    sprintf('Subband %d: expected %d rows, got %d', m, expected, size(c{m},1)));
            end
        end

        function testTDPeriodicBoundary(testCase)
            % 'per' extension runs without error and returns M subbands
            c = comp_filterbank_td(testCase.noise_real, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testTDZeroPaddingBoundary(testCase)
            % 'zpd' extension runs without error and returns M subbands
            c = comp_filterbank_td(testCase.noise_real, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'zpd');
            testCase.verifyEqual(length(c), testCase.M);
        end

        function testTDForwardInverseRoundtrip(testCase)
            % Analysis ('per') + synthesis ('per') with the SAME filters.
            % comp_ifilterbank_td signature: (c, g, a, Ls, offset, ext)
            c = comp_filterbank_td(testCase.noise_real, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');

            % For synthesis, offset convention differs: see comp_ifilterbank_td.
            % offset_s = -(Lh-1) places the reference at the last sample
            % (matching the adjoint of the causal analysis filter).
            offset_s = -(testCase.Lh - 1) * ones(testCase.M, 1);
            f_recon  = comp_ifilterbank_td(c, testCase.G_td, testCase.a, ...
                                           testCase.Ls, offset_s, 'per');

            % The energy should be non-trivially non-zero (not just a trivial pass)
            testCase.verifyGreaterThan(norm(f_recon), 0, ...
                'Synthesis output is zero for non-zero input');
            % Output has correct length
            testCase.verifyEqual(size(f_recon, 1), testCase.Ls);
        end

        function testTDShortSignal(testCase)
            % Short signal zero-padded to Ls: runs without error
            short_sig   = randn(16, 1);
            padded_short = postpad(short_sig, testCase.Ls);
            c = comp_filterbank_td(padded_short, testCase.G_td, ...
                                   testCase.a, testCase.offset, 'per');
            testCase.verifyEqual(length(c), testCase.M);
        end

    end

end
